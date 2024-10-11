#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import os
import yaml
import torch
import signal
import logging
from sklearn.model_selection import KFold
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from pytorch_lightning.plugins.environments import SLURMEnvironment

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

from core import datamodule, model, custom_logger, curvature
from utils import parameter_manager, model_loader

#--------------------------------
# Initialize: Training
#--------------------------------
  
def run(params):
    OMP_NUM_THREADS=1
    logging.debug("train.py() | running training")
    #logging.basicConfig(level=logging.DEBUG)

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
    n_splits = pm.params_datamodule['n_folds']
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=pm.seed_value)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        logging.info(f"Fold {fold_idx +1}/{n_splits}")
        
        # Initialize: The model for each fold
        model_instance = model_loader.select_model(pm, fold_idx)
        
        # Set up data module for this fold
        data_module.setup_fold(train_idx, val_idx)

        # Initialize: The logger
        logger = custom_logger.Logger(
            all_paths=pm.all_paths, name=f"{pm.model_id}_fold{fold_idx}", version=0
        )

        # Initialize:  PytorchLighting model checkpoint
        checkpoint_path = os.path.join(pm.path_root, pm.path_model, f"fold{fold_idx}")
        checkpoint_callback = ModelCheckpoint(dirpath = checkpoint_path)
    
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
                              callbacks = [checkpoint_callback, progress_bar],
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
                              callbacks = [checkpoint_callback, progress_bar],
                              num_sanity_val_steps = 1,
                              log_every_n_steps=1
                            )

        # Training
        trainer.fit(model_instance, data_module)
        
        # Testing
        trainer.test(model_instance, dataloaders=[data_module.val_dataloader(), data_module.train_dataloader()])
        
        # Analysis/Results and saving #TODO
        fold_results.append(model_instance.test_results)

    # Dump config for future reference
    yaml.dump(params, open(os.path.join(pm.path_root, f'{pm.path_results}/params.yaml'), 'w'))
    
    # Save results
    #results_path = os.path.join(pm.path_root, pm.path_results)
    #logging.debug(f"Results path: {results_path}")
    #custom_logger.save_results(fold_results, results_path, pm.model_id)
    
class CustomProgressBar(TQDMProgressBar):
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
        
    '''def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description(f"Fold {self.fold_idx + 1}/{self.total_folds} Training")
        return bar
    
    def on_train_epoch_start(self, trainer, pl_module):
        if self.train_progress_bar is not None:
            self.train_progress_bar.set_description(
                f"Fold {self.fold_idx + 1}/{self.total_folds} Training Epoch {trainer.current_epoch + 1}"
            )'''

    