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
import gc

sys.path.append('../')
import evaluation.evaluation as eval
import evaluation.inference as inference
import utils.model_loader as model_loader
from core import datamodule, custom_logger, train
from conf.schema import load_config

def plotting(conf, test_results, results_dir, fold_num=None, transfer=False):
    """
    The generation of a variety of plots and performance metrics
    """
    # plot training and validation loss from recorded loss.csv once
    if not os.path.exists(os.path.join(conf.paths.results, "loss_plots", "loss.pdf")):
        os.makedirs(os.path.join(conf.paths.results, "loss_plots"), exist_ok=True)
        print("Created directory: loss_plots")
        print("\nGenerating loss plots...")
        eval.plot_loss(conf, save_fig=True)
        
    # Create subdirectories for different types of results
    if transfer: # need to separate results if evaluating on all wavelengths
        wl = str(conf.data.wavelength).replace('.', '')
        results_dir = os.path.join(results_dir, f"eval_{wl}")
        os.makedirs(results_dir, exist_ok=True)
    metrics_dir = os.path.join(results_dir, "performance_metrics")
    output_subdir = "dft_plots" if conf.model.arch not in ["inverse", "NA"] else "radii_plots"
    output_dir = os.path.join(results_dir, output_subdir)
    flipbook_dir = os.path.join(results_dir, "flipbooks")
    directories = [metrics_dir, output_dir, flipbook_dir]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    # determine model type
    model_type = conf.model.arch
        
    # compute relevant metrics across folds
    if model_type != 'autoencoder':
        plot_mse = True if conf.model.arch not in ['mlp', 'cvnn', 'inverse', 'convTandem', 'NA'] else False
        print("\nComputing and saving metrics...")
        eval.metrics(test_results, dataset='train', save_fig=True, save_dir=results_dir, plot_mse=plot_mse)
        eval.metrics(test_results, dataset='valid', save_fig=True, save_dir=results_dir, plot_mse=plot_mse)

    if model_type == 'NA':
        for dataset in ['train', 'valid']:
            eval.summarize_na_results(test_results, dataset=dataset, save_dir=results_dir)
    
    if conf.model.arch not in ["inverse", "NA"]:    
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
    eval.plot_absolute_difference(conf, test_results, resub=True, sample_idx=10, 
                                  save_fig=True, save_dir=results_dir,
                                  arch=model_type, fold_num=fold_num)
    
    # visualize performance with animation
    if model_type not in ['autoencoder', 'cvnn', 'mlp', 'inverse', 'convTandem', 'NA']:
        print("\nGenerating field animations...")
        eval.animate_fields(test_results, dataset='valid', 
                            seq_len=conf.model.seq_len, save_dir=results_dir)
    
    print(f"\nEvaluation complete. All results saved to: {results_dir}")

def run(conf):
    # use current params to get results directory
    results_dir = conf.paths.results
    
    # setup new parameter manager based on saved parameters
    saved_conf = conf
        
    # Load model checkpoint
    model_path = os.path.join(results_dir, 'model.ckpt')
    model_instance = model_loader.select_model(saved_conf.model)
    if os.path.exists(model_path):
        model_instance.load_state_dict(torch.load(model_path, weights_only=False)['state_dict'])
    else:
        if saved_conf.model.arch == 'NA':
            print(f"Warning: checkpoint not found at {model_path}. Proceeding without loading weights for NA model (no trainable parameters).")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {model_path}")

    # empty logger so as not to mess with loss.csv
    logger = None

    # Checkpoint, EarlyStopping, ProgressBar
    checkpoint_path = saved_conf.paths.results
    filename = 'model'
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=filename,
        save_top_k=1,
        monitor='val_loss',
        mode='min' if saved_conf.model.objective_function == 'mse' else 'max',
        verbose=True
    )

    early_stopping = train.CustomEarlyStopping(
        monitor='val_loss',
        patience=saved_conf.trainer.patience,
        min_delta=saved_conf.trainer.min_delta,
        mode='min' if saved_conf.model.objective_function == 'mse' else 'max',
        verbose=True
    )

    progress_bar = train.CustomProgressBar()
    
    # ensure test results are empty so we can populate them
    for mode in ['train', 'valid']:
        for key in model_instance.test_results[mode].keys():
            model_instance.test_results[mode][key] =  []
    
    # setup the trainer
    trainer = train.configure_trainer(saved_conf, logger, checkpoint_callback, early_stopping, progress_bar)
    
    # determine if we're evaluating on a different wavelength
    transfer_eval = saved_conf.data.eval_wavelength != saved_conf.data.wavelength
    if transfer_eval:
        saved_conf.data.wavelength = saved_conf.data.eval_wavelength
    
    # init datamodule
    data_module = datamodule.select_data(saved_conf)
    data_module.prepare_data()
    data_module.setup(stage='fit')
    
    if (saved_conf.trainer.cross_validation):
        with open(os.path.join(results_dir, "split_info.yaml"), 'r') as f:
            split_info = yaml.safe_load(f)
        train_idx = split_info["train_idx"]
        val_idx = split_info["val_idx"]
        data_module.setup_fold(train_idx, val_idx)
    else: # cross validation was not conducted
        data_module.setup_og()
    
    # perform testing
    trainer.test(model_instance, dataloaders=[data_module.val_dataloader(), data_module.train_dataloader()])
    
    # evaluate
    plotting(saved_conf, model_instance.test_results, 
            results_dir, transfer=transfer_eval)
    
    
