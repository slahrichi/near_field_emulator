# actual driver for evaluation - produces metrics/plots/etc/
import torch
import matplotlib.pyplot as plt
import sys
import os
import yaml
import numpy as np
from sklearn.model_selection import KFold
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping

sys.path.append('../')
import evaluation.evaluation as eval
import evaluation.inference as inference
import utils.parameter_manager as parameter_manager
import utils.model_loader as model_loader
from utils.mapping import get_model_type
from core import datamodule, custom_logger, train

def plotting(pm, test_results, results_dir, fold_num=None):
    """
    The generation of a variety of plots and performance metrics
    """
    # Create subdirectories for different types of results
    loss_dir = os.path.join(results_dir, "loss_plots")
    metrics_dir = os.path.join(results_dir, "performance_metrics")
    dft_dir = os.path.join(results_dir, "dft_plots")
    flipbook_dir = os.path.join(results_dir, "flipbooks")
    
    for directory in [loss_dir, metrics_dir, dft_dir, flipbook_dir]:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # plot training and validation loss
    print("\nGenerating loss plots...")
    eval.plot_loss(pm, save_fig=True, save_dir=results_dir)

    # determine model type
    model_type = get_model_type(pm.arch)
        
    # compute relevant metrics across folds
    if model_type != 'autoencoder':
        print("\nComputing and saving metrics...")
        eval.print_metrics(test_results, dataset='train', save_fig=True, save_dir=results_dir)
        eval.print_metrics(test_results, dataset='valid', save_fig=True, save_dir=results_dir)
    
    # visualize performance with DFT fields
    print("\nGenerating DFT field plots...")
    eval.plot_dft_fields(test_results, resub=True, sample_idx=10, save_fig=True, 
                         save_dir=results_dir, arch=model_type, format='polar',
                         fold_num=fold_num)
    if model_type == 'mlp':
        eval.plot_dft_fields(test_results, resub=True, sample_idx=10, save_fig=True, 
                             save_dir=results_dir, arch=model_type, format='cartesian',
                             fold_num=fold_num)
    #if model_type == 'lstm' or model_type == 'convlstm':
    eval.plot_absolute_difference(test_results, resub=True, sample_idx=10, 
                                  save_fig=True, save_dir=results_dir,
                                  arch=model_type, fold_num=fold_num)
    
    # visualize performance with animation
    if model_type != 'autoencoder' and model_type != 'mlp':
        print("\nGenerating field animations...")
        eval.animate_fields(test_results, dataset='valid', 
                            seq_len=pm.seq_len, save_dir=results_dir)
    
    print(f"\nEvaluation complete. All results saved to: {results_dir}")

def run(params):
    # use current params to get results directory
    pm_temp = parameter_manager.Parameter_Manager(params=params)
    results_dir = os.path.join(pm_temp.path_root, pm_temp.path_results)
    
    # setup new parameter manager based on saved parameters
    model_params = yaml.load(open(os.path.join(results_dir, 'params.yaml')), 
                             Loader=yaml.FullLoader).copy()
    pm = parameter_manager.Parameter_Manager(params=model_params)
        
    # Load model checkpoint
    model_path = os.path.join(results_dir, 'model.ckpt')
    model_instance = model_loader.select_model(pm)
    model_instance.load_state_dict(torch.load(model_path)['state_dict'])
    
    # init datamodule
    data_module = datamodule.select_data(pm.params_datamodule)
    data_module.prepare_data()
    data_module.setup(stage='fit')
    
    if (pm.cross_validation):
        with open(os.path.join(results_dir, "split_info.yaml"), 'r') as f:
            split_info = yaml.safe_load(f)
        train_idx = split_info["train_idx"]
        val_idx = split_info["val_idx"]
        data_module.setup_fold(train_idx, val_idx)
    else: # cross validation was not conducted
        data_module.setup_og()
        
    model_instance = model_loader.select_model(pm)
    # empty logger so as not to mess with loss.csv
    logger = None

    # Checkpoint, EarlyStopping, ProgressBar
    checkpoint_path = os.path.join(pm.path_root, pm.path_results)
    filename = 'model'
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=filename,
        save_top_k=1,
        monitor='val_loss',
        mode='min' if pm.params['objective_function'] == 'mse' else 'max',
        verbose=True
    )

    early_stopping = train.CustomEarlyStopping(
        monitor='val_loss',
        patience=pm.patience,
        min_delta=pm.min_delta,
        mode='min' if pm.params['objective_function'] == 'mse' else 'max',
        verbose=True
    )

    progress_bar = train.CustomProgressBar()
    
    # ensure test results are empty so we can populate them
    model_instance.test_results = {'train': {'nf_pred': [], 'nf_truth': []},
                                    'valid': {'nf_pred': [], 'nf_truth': []}}
    
    # setup the Trainer and perform testing
    trainer = train.configure_trainer(pm, logger, checkpoint_callback, early_stopping, progress_bar)
    trainer.test(model_instance, dataloaders=[data_module.val_dataloader(), data_module.train_dataloader()])
    
    # evaluate
    plotting(pm, model_instance.test_results, results_dir)
    
    
    
