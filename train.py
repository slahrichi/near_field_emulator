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
from sklearn.model_selection import KFold
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from pytorch_lightning.plugins.environments import SLURMEnvironment

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

from core import datamodule, model, custom_logger, curvature
from utils import parameter_manager, model_loader

# debugging
#logging.basicConfig(level=logging.DEBUG)

#--------------------------------
# Initialize: Training
#--------------------------------

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

def train(params):
    OMP_NUM_THREADS=1
    logging.debug("train.py() | running training")

    # Initialize: Parameter manager
    pm = parameter_manager.Parameter_Manager(params=params)
          
    # Initialize: Seeding
    if(pm.seed_flag):
        seed_everything(pm.seed_value, workers = True)

    # Initialize: The datamodule
    data_module = datamodule.select_data(pm.params_datamodule)
    
    # prepare data and setup datamodule
    data_module.prepare_data()
    data_module.setup(stage='fit')
    
    # access the full dataset
    full_dataset = data_module.dataset
    
    # Initialize: K-Fold Cross Validation
    if(pm.cross_validation):
        n_splits = pm.params_datamodule['n_folds']
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=pm.seed_value)
    else: # 80/20 split
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=pm.seed_value)
    
    fold_results = []
    best_val_loss = float('inf')
    best_model_path = None
    
    # Dump config for future reference
    yaml.dump(params, open(os.path.join(pm.path_root, f'{pm.path_results}/params.yaml'), 'w'))
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        logging.info(f"Fold {fold_idx +1}/{n_splits}")
        
        if fold_idx > 0:
            clear_memory()
            if(not pm.cross_validation): # only run first fold
                break
        
        # Initialize: new fold
        model_instance = model_loader.select_model(pm, fold_idx)
        data_module.setup_fold(train_idx, val_idx)

        # Initialize: The logger
        logger = custom_logger.Logger(
            all_paths=pm.all_paths,
            name=f"{pm.model_id}_fold{fold_idx + 1}", 
            version=0, 
            fold_idx=fold_idx
        )

        # Initialize:  PytorchLighting model checkpoint
        checkpoint_path = os.path.join(pm.path_root, pm.path_results)
        if pm.cross_validation:
            filename = f'model_fold{fold_idx + 1}'
        else:
            filename = 'model'
        checkpoint_callback = ModelCheckpoint(
            dirpath = checkpoint_path,
            filename=filename,
            save_top_k=1,
            monitor='val_loss',
            mode='min',
            verbose=True
        )
        
        early_stopping = CustomEarlyStopping(
            monitor='val_loss',
            patience=pm.patience,
            min_delta=pm.min_delta,
            mode='min',
            verbose=True
        )
    
        logging.debug(f'Checkpoint path: {checkpoint_path}')
        logging.debug('Setting matmul precision to HIGH')
        torch.set_float32_matmul_precision('high')
        
        progress_bar = CustomProgressBar(fold_idx, n_splits)

        # Initialize: PytorchLightning Trainer
        if(pm.gpu_flag and torch.cuda.is_available()):
            logging.debug("Training with GPUs")
            trainer = Trainer(logger = logger, 
                              accelerator = "cuda", 
                              num_nodes = 1, 
                              devices = pm.gpu_list,
                              max_epochs = pm.num_epochs, 
                              deterministic=True,
                              enable_progress_bar=True,
                              enable_model_summary=True,
                              default_root_dir = pm.path_root,
                              callbacks = [checkpoint_callback, early_stopping, progress_bar],
                              check_val_every_n_epoch = pm.valid_rate,
                              num_sanity_val_steps = 1,
                              log_every_n_steps=1
                              #, plugins = [SLURMEnvironment(requeue_signal=signal.SIGHUP),]
                            )
        else:
            logging.debug("Training with CPUs")
            trainer = Trainer(logger = logger,
                              accelerator = "cpu",
                              max_epochs = pm.num_epochs,
                              deterministic=True,
                              enable_progress_bar = True,
                              enable_model_summary = True,
                              default_root_dir = pm.path_root, 
                              check_val_every_n_epoch = pm.valid_rate,
                              callbacks = [checkpoint_callback, early_stopping, progress_bar],
                              num_sanity_val_steps = 1,
                              log_every_n_steps=1
                            )

        # Training
        trainer.fit(model_instance, data_module)
        
        # note validation loss of most recent fold
        current_val_loss = checkpoint_callback.best_model_score.item()
        
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_path = checkpoint_callback.best_model_path
            logging.info(f"New best model found in fold {fold_idx + 1} with validation loss: {best_val_loss:.6f}")
        
        # Testing
        if(pm.include_testing):
            trainer.test(model_instance, dataloaders=[data_module.val_dataloader(), data_module.train_dataloader()])
        
            # Analysis/Results and saving #TODO
            fold_results.append(model_instance.test_results)
        
    # After all folds complete, grab the best one, save it, and delete others
    # to save space
    if best_model_path:
        results_dir = os.path.join(pm.path_root, pm.path_results)
        os.makedirs(results_dir, exist_ok=True)
        checkpoint_path = os.path.join(results_dir, 'model.ckpt')
        
        # save best model
        best_model = torch.load(best_model_path)
        torch.save(best_model, checkpoint_path)
        logging.info(f"Saved best overall model to {checkpoint_path}")
        
        # Clean up temporary checkpoints
        for fold in range(n_splits):
            temp_fold_ckpt = os.path.join(pm.path_root, pm.path_results, f"model_fold{fold + 1}.ckpt")
            if os.path.exists(temp_fold_ckpt): # remove extraneous checkpoints
                os.remove(temp_fold_ckpt)
    
class CustomProgressBar(TQDMProgressBar):
    """Custom progress bar that adds fold information"""
    def __init__(self, fold_idx, total_folds):
        super().__init__()
        self.fold_idx = fold_idx
        self.total_folds = total_folds
        
    def get_metrics(self, trainer, model):
        # Get the base metrics
        base_metrics = super().get_metrics(trainer, model)
        # Add fold information
        base_metrics["fold"] = f"{self.fold_idx + 1}/{self.total_folds}"
        return base_metrics
    
class CustomEarlyStopping(EarlyStopping):
    """Custom Early Stopping class for controlling the training loop;  
    ensuring that we terminate the model after it stops improving.
    """
    def __init__(self, monitor='val_loss', patience=5, min_delta=0.01, mode='min', verbose=True):
        super().__init__(monitor=monitor, patience=patience, min_delta=min_delta, mode=mode, verbose=verbose)
        self.wait_count = 0
        self.initial_score = None
        self.last_epoch_processed = -1
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # only once per epoch
        if trainer.current_epoch == self.last_epoch_processed:
            return
        self.last_epoch_processed = trainer.current_epoch
        
        # Get the current score
        current_score = trainer.callback_metrics[self.monitor]
        if current_score is None:
            # Metric not available; do nothing
            return
        
        # Ensure current_score is a float
        if isinstance(current_score, torch.Tensor):
            current_score = current_score.item()
        
        # set the initial score
        if self.initial_score is None:
            self.initial_score = current_score
            self.wait_count = 0
            
        # Calculate total improvement over the patience period
        if self.mode == 'min':
            total_improvement = self.initial_score - current_score
            # total_improvement should be negative to match min_delta    
            total_improvement = -total_improvement
        else:  # mode == 'max'
            total_improvement = current_score - self.initial_score
            
        # Ensure total_improvement is a float
        if isinstance(total_improvement, torch.Tensor):
            total_improvement = total_improvement.item()
            
        #if self.verbose:
        #    print(f"\nEpoch {trainer.current_epoch}: total_improvement = {total_improvement:.5f}, min_delta = {self.min_delta}\n")
            
        # check if total_improvement exceeds min_delta
        if total_improvement <= self.min_delta:
            # reset counters
            self.initial_score = current_score
            self.wait_count = 0
            if self.verbose:
                print(f"\nEpoch {trainer.current_epoch}: Improvement of {total_improvement:.5f} observed; continuing training.\n")
        else: # didn't improve enough
            self.wait_count += 1
            if self.verbose:
                print(f"\nEpoch {trainer.current_epoch}: No sufficient improvement; wait_count = {self.wait_count}/{self.patience}\n")
            if self.wait_count >= self.patience:
                if self.verbose:
                    print(f"\nEarlyStopping at epoch {trainer.current_epoch}: {self.monitor} did not improve by at least {self.min_delta} over the last {self.patience} epochs.\n")
                trainer.should_stop = True
    