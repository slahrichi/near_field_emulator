# actual driver for evaluation - produces metrics/plots/etc/
import torch
import matplotlib.pyplot as plt
import sys
import os
import yaml

sys.path.append('../')
import evaluation.evaluation as eval
import utils.parameter_manager as parameter_manager

def eval_model(params):
    
    # use current params to get results directory
    pm_temp = parameter_manager.Parameter_Manager(params=params)
    results_dir = os.path.join(pm_temp.path_root, pm_temp.path_results)
    
    # setup new parameter manager based on saved parameters
    model_params = yaml.load(open(os.path.join(results_dir, 'params.yaml')), 
                             Loader=yaml.FullLoader).copy()
    pm = parameter_manager.Parameter_Manager(params=model_params)
    
    # Create subdirectories for different types of results
    loss_dir = os.path.join(results_dir, "loss_plots")
    metrics_dir = os.path.join(results_dir, "performance_metrics")
    dft_dir = os.path.join(results_dir, "dft_plots")
    flipbook_dir = os.path.join(results_dir, "flipbooks")
    
    for directory in [loss_dir, metrics_dir, dft_dir, flipbook_dir]:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # get results from all folds
    fold_results = eval.get_all_results(results_dir, pm.n_folds)
    
    # plot training and validation loss
    print("Generating loss plots...")
    eval.plot_loss(pm, fold_results, save_fig=True, save_dir=results_dir)
    
    # compute relevant metrics across folds
    print("Computing and saving metrics...")
    eval.print_metrics(fold_results, dataset='train', save_fig=True, save_dir=results_dir)
    eval.print_metrics(fold_results, dataset='valid', save_fig=True, save_dir=results_dir)
    
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
    
    # visualize performance with DFT fields
    print("Generating DFT field plots...")
    eval.plot_dft_fields(fold_results, plot_type='best', 
                         save_fig=True, save_dir=results_dir,
                         arch=model_type, format='polar')
    
    # visualize performance with animation
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
    
    
