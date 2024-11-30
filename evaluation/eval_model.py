# actual driver for evaluation - produces metrics/plots/etc/
import torch
import matplotlib.pyplot as plt
import sys
import os
import yaml
import numpy as np
from sklearn.model_selection import KFold
from pytorch_lightning import Trainer

sys.path.append('../')
import evaluation.evaluation as eval
import evaluation.inference as inference
import utils.parameter_manager as parameter_manager
import utils.model_loader as model_loader
from core import datamodule, custom_logger

def eval_model(params):
    
    # use current params to get results directory
    pm_temp = parameter_manager.Parameter_Manager(params=params)
    if pm_temp.model_id == 'ae-v1':
        results_dir = os.path.join(pm_temp.path_root, pm_temp.path_pretrained_ae)
    else:
        results_dir = os.path.join(pm_temp.path_root, pm_temp.path_results)
    
    # setup new parameter manager based on saved parameters
    model_params = yaml.load(open(os.path.join(results_dir, 'params.yaml')), 
                             Loader=yaml.FullLoader).copy()
    pm = parameter_manager.Parameter_Manager(params=model_params)
    
    if not pm_temp.include_testing:
        # need to perform testing
            # Load model and data
        model_path = os.path.join(results_dir, 'model.ckpt')
        model_instance = model_loader.select_model(pm)
        model_instance.load_state_dict(torch.load(model_path)['state_dict'])
        
        # init datamodule
        data_module = datamodule.select_data(pm.params_datamodule)
        data_module.prepare_data()
        data_module.setup(stage='fit')
        
        # Initialize: K-Fold Cross Validation
        if(pm.cross_validation):
            n_splits = pm.params_datamodule['n_folds']
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=pm.seed_value)
        else: # 80/20 split
            kf = KFold(n_splits=5, shuffle=True, random_state=pm.seed_value)
        train_idx, val_idx = next(kf.split(range(len(data_module.dataset))))
        data_module.setup_fold(train_idx, val_idx)
        
        # Initialize: The logger
        logger = custom_logger.Logger(
            all_paths=pm.all_paths,
            name=f"{pm.model_id}_fold{0 + 1}", 
            version=0, 
            fold_idx=0
        )
        
        # Setup trainer for testing only
        trainer = Trainer(
            accelerator="cuda" if pm.gpu_flag and torch.cuda.is_available() else "cpu",
            devices=pm.gpu_list if pm.gpu_flag and torch.cuda.is_available() else None,
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=logger
        )
        
        # Run test on validation and training sets
        trainer.test(model_instance, dataloaders=[
            data_module.val_dataloader(),
            data_module.train_dataloader()
        ])
          
    # Create subdirectories for different types of results
    loss_dir = os.path.join(results_dir, "loss_plots")
    metrics_dir = os.path.join(results_dir, "performance_metrics")
    dft_dir = os.path.join(results_dir, "dft_plots")
    flipbook_dir = os.path.join(results_dir, "flipbooks")
    
    for directory in [loss_dir, metrics_dir, dft_dir, flipbook_dir]:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # get results from all folds
    if pm.cross_validation:
        fold_results = eval.get_all_results(results_dir, pm.n_folds)
    else:
        fold_results = eval.get_all_results(results_dir, 1)
    
    # plot training and validation loss
    print("Generating loss plots...")
    eval.plot_loss(pm, fold_results, save_fig=True, save_dir=results_dir)

    # determine model type
    if pm.experiment == 1:
        model_type = 'autoencoder'
    else:
        if pm.arch == 0:
            model_type = 'mlp'
        elif pm.arch == 1 or pm.arch == 2:
            model_type = 'lstm' if pm.arch == 1 else 'convlstm'
        else:
            raise ValueError("Model type not recognized")
        
    # compute relevant metrics across folds
    if model_type != 'autoencoder':
        print("Computing and saving metrics...")
        eval.print_metrics(fold_results, dataset='train', save_fig=True, save_dir=results_dir)
        eval.print_metrics(fold_results, dataset='valid', save_fig=True, save_dir=results_dir)
    
    # visualize performance with DFT fields
    print("Generating DFT field plots...")
    eval.plot_dft_fields(fold_results, plot_type='best', 
                        save_fig=True, save_dir=results_dir,
                        arch=model_type, format='polar')
    
    # visualize performance with animation
    if model_type != 'autoencoder':
        print("Generating field animations...")
        eval.animate_fields(fold_results, dataset='valid', 
                            seq_len=pm.seq_len, save_dir=results_dir)
    
    print(f"\nEvaluation complete. All results saved to: {results_dir}")
    # List all generated files
    print("\nGenerated files:")
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith(('.pdf', '.png', '.gif')):
                print(f"- {os.path.join(root, file)}")
    
    # After all plots and metrics are generated, clean up the large results files
    print("\nCleaning up large results files...")
    for mode in ['train', 'valid']:
        for fold in range(pm.n_folds if pm.cross_validation else 1):
            fold_suffix = f"_fold{fold+1}"
            results_file = os.path.join(results_dir, f'{mode}_info', f'results{fold_suffix}.pkl')
            if os.path.exists(results_file):
                os.remove(results_file)
                print(f"Removed: {results_file}")
    
