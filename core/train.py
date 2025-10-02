#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import os
import yaml
import torch
import signal
import gc
import logging
import shutil
import sys
from sklearn.model_selection import KFold
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from pytorch_lightning.plugins.environments import SLURMEnvironment

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
sys.path.append('../')
from core import datamodule, custom_logger
from utils import model_loader, mapping
from conf.schema import load_config

# debugging
#logging.basicConfig(level=logging.DEBUG)

#--------------------------------
# Utilities
#--------------------------------

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    
class CustomProgressBar(TQDMProgressBar):
    """Custom progress bar that adds fold information"""
    def __init__(self, fold_idx=None, total_folds=None):
        super().__init__()
        self.fold_idx = fold_idx
        self.total_folds = total_folds
        
    def get_metrics(self, trainer, model):
        base_metrics = super().get_metrics(trainer, model)
        if self.fold_idx is not None and self.total_folds is not None:
            base_metrics["fold"] = f"{self.fold_idx + 1}/{self.total_folds}"
        return base_metrics
    
class CustomEarlyStopping(EarlyStopping):
    """Custom Early Stopping class to provide more verbose logging."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _log_info(self, trainer, reason: str, log_rank_zero_only: bool = False):
        super()._log_info(trainer, reason, log_rank_zero_only)
        # The parent class's logging is already quite informative.
        # You can add more custom logging here if needed.

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if self.verbose and not trainer.should_stop:
            current_score = trainer.callback_metrics.get(self.monitor)
            if current_score is not None:
                print(f"Epoch {trainer.current_epoch}: Monitored metric {self.monitor} = {current_score:.6f}. "
                      f"Patience count = {self.wait_count}/{self.patience}.")
           
def configure_trainer(conf, logger, checkpoint_callback, early_stopping, progress_bar):
    """Create and return a configured Trainer instance."""
    trainer_kwargs = {
        'logger': logger,
        'max_epochs': conf.trainer.num_epochs,
        'deterministic': True,
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'default_root_dir': conf.paths.root,
        'callbacks': [checkpoint_callback, early_stopping, progress_bar],
        'check_val_every_n_epoch': conf.trainer.valid_rate,
        'num_sanity_val_steps': 1,
        'log_every_n_steps': 1,
        'inference_mode': False
    }

    if torch.cuda.is_available():
        trainer_kwargs.update({
            'accelerator': 'cuda',
        })
        logging.debug("Training with GPUs")
    else:
        trainer_kwargs.update({
            'accelerator': 'cpu'
        })
        logging.debug("Training with CPUs")

    return Trainer(**trainer_kwargs)

def save_best_model(conf, best_model_path, n_splits=None):
    """Save the best model checkpoint and clean up temporary ones."""
    if best_model_path:
        results_dir = conf.paths.results
        os.makedirs(results_dir, exist_ok=True)
        checkpoint_path = os.path.join(results_dir, 'model.ckpt')
        
        best_model = torch.load(best_model_path, weights_only=False)
        torch.save(best_model, checkpoint_path)
        logging.info(f"Saved best overall model to {checkpoint_path}")
        
        # If cross-validation was used, remove fold checkpoints
        if n_splits is not None:
            for fold in range(n_splits):
                temp_fold_ckpt = os.path.join(conf.paths.results, f"model_fold{fold + 1}.ckpt")
                if os.path.exists(temp_fold_ckpt):
                    os.remove(temp_fold_ckpt)
                    
def record_split_info(fold_idx, train_idx, val_idx, results_dir):
    """
    Records index info for the best performing fold for later reference during evaluation.
    """
    fold_num = fold_idx + 1
    split_info = {
        "best_fold": fold_num,
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist()
    }
    with open(os.path.join(results_dir, "split_info.yaml"), 'w') as f:
        yaml.dump(split_info, f)


#--------------------------------
# Training Functions
#--------------------------------
                  
def train_once(conf, data_module):
    """
    Train without cross-validation, utilizing the train/valid split originally
    established during data preprocessing (should be 80/20) 
    The split: core/preprocess_data.py --> separate_datasets()
    Tagged in core/datamodule.py --> load_pickle_data()
    """
    data_module.setup_og()

    model_instance = model_loader.select_model(conf.model)
    logger = custom_logger.Logger(
        all_paths=conf.paths,
        name=conf.model.model_id,
        version=0
    )

    # Checkpoint and EarlyStopping
    checkpoint_path = conf.paths.results
    filename = 'model'
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=filename,
        save_top_k=1,
        monitor='val_loss',
        mode='min' if conf.model.objective_function == 'mse' else 'max',
        verbose=True
    )

    early_stopping = CustomEarlyStopping(
        monitor='val_loss',
        patience=conf.trainer.patience,
        min_delta=conf.trainer.min_delta,
        mode='min' if conf.model.objective_function == 'mse' else 'max',
        verbose=True
    )

    progress_bar = CustomProgressBar()

    trainer = configure_trainer(conf, logger, checkpoint_callback, early_stopping, progress_bar)

    # Train
    trainer.fit(model_instance, data_module)
    
    # Save best model
    best_model_path = checkpoint_callback.best_model_path
    save_best_model(conf, best_model_path, n_splits=None)

    # Test if needed
    if conf.trainer.include_testing:
        trainer.test(model_instance, dataloaders=[data_module.val_dataloader(), data_module.train_dataloader()])
    else:
        base_path = os.path.dirname(best_model_path)
        if os.path.exists(os.path.join(base_path, 'train_info')):
            # remove train_info and valid_info dirs from base_path #TODO: cleaner to have this in logger
            shutil.rmtree(os.path.join(base_path, 'train_info'))
            shutil.rmtree(os.path.join(base_path, 'valid_info'))


def train_with_cross_validation(conf, data_module):
    """Train using K-Fold Cross Validation."""
    full_dataset = data_module.dataset
    n_splits = conf.data.n_folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=conf.seed_value)

    best_val_loss = float('-inf') if conf.model.objective_function == 'psnr' else float('inf')
    best_model_path = None

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        logging.info(f"Fold {fold_idx + 1}/{n_splits}")
        if fold_idx > 0:
            clear_memory()

        # Add data source info to model config
        conf.model.source = conf.data.source
        conf.model.num_projections = conf.data.num_projections

        # Pass the model config to the model
        model_instance = model_loader.select_model(conf.model, fold_idx)
        data_module.setup_fold(train_idx, val_idx)

        logger = custom_logger.Logger(
            all_paths=conf.paths,
            name=f"{conf.model.model_id}_fold{fold_idx + 1}", 
            version=0, 
            fold_idx=fold_idx
        )

        checkpoint_path = conf.paths.results
        filename = f'model_fold{fold_idx + 1}'
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_path,
            filename=filename,
            save_top_k=1,
            monitor='val_loss',
            mode='min' if conf.model.objective_function == 'mse' else 'max',
            verbose=True
        )
        
        early_stopping = CustomEarlyStopping(
            monitor='val_loss',
            patience=conf.trainer.patience,
            min_delta=conf.trainer.min_delta,
            mode='min' if conf.model.objective_function == 'mse' else 'max',
            verbose=True
        )

        progress_bar = CustomProgressBar(fold_idx, n_splits)
        trainer = configure_trainer(conf, logger, checkpoint_callback, early_stopping, progress_bar)

        # Training
        trainer.fit(model_instance, data_module)

        current_val_loss = checkpoint_callback.best_model_score.item()

        if conf.model.objective_function == 'psnr':
            if current_val_loss > best_val_loss:
                best_val_loss = current_val_loss
                best_model_path = checkpoint_callback.best_model_path
                logging.info(f"New best model found in fold {fold_idx + 1} with PSNR: {best_val_loss:.6f}")
                record_split_info(fold_idx, train_idx, val_idx, os.path.dirname(best_model_path))
        else:
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_model_path = checkpoint_callback.best_model_path
                logging.info(f"New best model found in fold {fold_idx + 1} with val loss: {best_val_loss:.6f}")
                record_split_info(fold_idx, train_idx, val_idx, os.path.dirname(best_model_path))

        # Testing if needed
        if conf.trainer.include_testing:
            trainer.test(model_instance, dataloaders=[data_module.val_dataloader(), data_module.train_dataloader()])
            fold_results.append(model_instance.test_results)
        else:
            base_path = os.path.dirname(best_model_path)
            # remove train_info and valid_info dirs from base_path #TODO: cleaner to have this in logger
            shutil.rmtree(os.path.join(base_path, 'train_info'))
            shutil.rmtree(os.path.join(base_path, 'valid_info'))

    # After all folds, save the best model
    save_best_model(conf, best_model_path, n_splits)

#--------------------------------
# Main Training Entry Point
#--------------------------------

def run(conf):
    logging.debug("train.py() | running training")

    # Dump config for future reference
    os.makedirs(conf.paths.results, exist_ok=True)
    #os.makedirs(os.path.join(conf.paths.results, conf.model.arch, f"model_{conf.model.model_id}"), exist_ok=True)
    # Initialize: The datamodule
    data_module = datamodule.select_data(conf)
    data_module.prepare_data()
    data_module.setup(stage='fit')
     
    # run training
    if(conf.trainer.cross_validation):
        train_with_cross_validation(conf, data_module)
    else: # run once
        train_once(conf, data_module)
    shutil.copy('/app/near_field_emulator/conf/config.yaml', os.path.join(conf.paths.results, 'params.yaml'))
    
