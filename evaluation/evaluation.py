import pickle
import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.font_manager import FontProperties
import matplotlib.cm as cm
import scipy.stats as stats
import yaml

fontsize = 8
font = FontProperties()
colors = ['darkgreen','purple','#4e88d9'] 

path_results = "/develop/results/"
model_identifier = ""

def is_csv_empty(path):
    try:
        df = pd.read_csv(path)
        if df.empty:
            return True
    except pd.errors.EmptyDataError:
        return True

    return False

def load_losses(fold_num, folder_path):
    """Loads losses for a specific fold"""
    path = os.path.join(folder_path, f"loss_fold{fold_num+1}.csv")
    if is_csv_empty(path):
        print("Empty CSV file")
        return False
    losses = pd.read_csv(path)
    return losses


def get_results(folder_path, n_folds, fold_num=1):

    train_path = os.path.join(folder_path, "train_info")
    valid_path = os.path.join(folder_path, "valid_info")

    losses = load_losses(fold_num, folder_path)

    train_results = pickle.load(open(os.path.join(train_path, f"fold{fold_num}", f"results_fold{fold_num}.pkl"), "rb"))
    valid_results = pickle.load(open(os.path.join(valid_path, f"fold{fold_num}", f"results_fold{fold_num}.pkl"), "rb"))
    
    return losses, train_results, valid_results

def get_all_results(folder_path, n_folds):
    fold_results = []

    # Loop over all folds
    for fold_num in range(n_folds):
        # Define paths for the current fold
        train_path = os.path.join(folder_path, "train_info", f"fold{fold_num+1}", f"results_fold{fold_num+1}.pkl")
        valid_path = os.path.join(folder_path, "valid_info", f"fold{fold_num+1}", f"results_fold{fold_num+1}.pkl")
        
        # Load the train and valid results for this fold
        train_results = pickle.load(open(train_path, "rb"))
        valid_results = pickle.load(open(valid_path, "rb"))

        # Load the losses for this fold (assuming your losses are saved in a similar way)
        losses = load_losses(fold_num, folder_path)

        fold_results.append({
            'train': train_results,    # Contains 'nf_truth' and 'nf_pred' for training
            'valid': valid_results,    # Contains 'nf_truth' and 'nf_pred' for validation
            'losses': losses           # Contains 'epoch', 'train_loss', and 'val_loss'
        })

    return fold_results

def gather_info(folder_path):
    excess = os.path.join(path_results, "model_cai_")
    file_path = os.path.join(folder_path, "params.yaml")
    if os.path.isfile(file_path):

        with open(file_path, 'r') as file:
            yaml_content = yaml.safe_load(file)
            key_dict = {}    
            lr = key_dict['lr'] = yaml_content['learning_rate']
            optimizer = key_dict['optimizer'] = yaml_content['optimizer']
            batch_size = key_dict['batch_size'] = yaml_content['batch_size']
            mlp_real = key_dict['mlp_real'] = yaml_content['mlp_real']
            mlp_imag = key_dict['mlp_imag'] = yaml_content['mlp_imag']
            n_folds = key_dict['n_folds'] = yaml_content['n_folds']
            key_dict['title'] = folder_path.replace(excess, "")

            return key_dict

def plot_loss(model_info, fold_results, min_list, max_list, save_fig=False, save_dir=None):
    
    losses = [fold['losses'] for fold in fold_results]
    title = model_info['title'].split("/")[-1]
    lr = model_info['lr']
    optimizer = model_info['optimizer']
    batch_size = model_info['batch_size']
    mlp_layers = model_info['mlp_real']['layers']
    model_identifier = f'{title} - lr: {lr}, optimizer: {optimizer}, batch_size: {batch_size}, mlp_layers: {mlp_layers}'
    #model_identifier = f'{title} - lr: {lr}, optimizer: {optimizer}, batch_size: {batch_size}'
    
    plt.style.use("ggplot")
    
    # Create two subplots: one for training and one for validation
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))  # 1 row, 2 columns
    
    train_loss_dfs = []
    val_loss_dfs = []
    
    for fold_idx, loss_df in enumerate(losses):
        loss_df = loss_df.set_index('epoch')
        train_loss = loss_df['train_loss'].dropna()
        val_loss = loss_df['val_loss'].dropna()
        
        train_loss.name = f'Fold_{fold_idx+1}'
        val_loss.name = f'Fold_{fold_idx+1}'
        
        train_loss_dfs.append(train_loss)
        val_loss_dfs.append(val_loss)
        
    # align on the epoch index
    train_loss_df = pd.concat(train_loss_dfs, axis=1)
    val_loss_df = pd.concat(val_loss_dfs, axis=1)
    
    # compute across folds and ignore NaN
    mean_train_loss = train_loss_df.mean(axis=1, skipna=True)
    std_train_loss = train_loss_df.std(axis=1, skipna=True)
    
    mean_val_loss = val_loss_df.mean(axis=1, skipna=True)
    std_val_loss = val_loss_df.std(axis=1, skipna=True)

    # Extract the epochs
    #epochs = losses[0]["epoch"]

    # Plot training loss with std deviation
    ax[0].plot(mean_train_loss.index, mean_train_loss.values, color='red', label=f'Training Mean')
    ax[0].fill_between(mean_train_loss.index, 
                       mean_train_loss.values - std_train_loss.values, 
                       mean_train_loss.values + std_train_loss.values, 
                       color='red', alpha=0.3, label='Training Std Dev')
    ax[0].set_ylabel("Loss", fontsize=10)
    ax[0].set_xlabel("Epoch", fontsize=10)
    ax[0].set_title(f"Training Loss", fontsize=12)
    ax[0].set_ylim([min_list[0], max_list[0]])
    ax[0].legend()
    
    # Plot validation loss with std deviation
    ax[1].plot(mean_val_loss.index, mean_val_loss.values, color='blue', label=f'Validation Mean')
    ax[1].fill_between(mean_val_loss.index, 
                       mean_val_loss.values - std_val_loss.values, 
                       mean_val_loss.values + std_val_loss.values, 
                       color='blue', alpha=0.3, label='Validation Std Dev')
    ax[1].set_ylabel("Loss", fontsize=10)
    ax[1].set_xlabel("Epoch", fontsize=10)
    ax[1].set_title(f"Validation Loss", fontsize=12)
    ax[1].set_ylim([min_list[1], max_list[1]])
    ax[1].legend()
    
    fig.suptitle(model_identifier)
    fig.tight_layout()
    
    # Save the plot if requested
    if save_fig:
        if save_dir is None:
            save_dir = os.getcwd()
        loss_plots_dir = os.path.join(save_dir, "loss_plots")
        os.makedirs(loss_plots_dir, exist_ok=True)
        fig.savefig(os.path.join(loss_plots_dir, f'{title}.pdf'))
        print(f"Figure saved to {os.path.join(loss_plots_dir, f'{title}.pdf')}")

def plot_dft_fields(fold_results, fold_idx=None, plot_type="best", sample_idx=0, save_fig=False, save_dir=None):
    """
    Parameters:
    - fold_results: List of dictionaries containing train and valid results for each fold
    - fold_idx: Optional, if you want to visualize a specific fold by index
    - plot_type: "best" to plot best-performing fold, "worst" to plot worst-performing fold, or "specific" if fold_idx is provided
    """
    def plot_single_set(results, title, save_path, sample_idx):
        # extract the specific sample from the results
        truth_real = results['nf_truth'][sample_idx, 0, :, :]
        truth_imag = results['nf_truth'][sample_idx, 1, :, :]
        pred_real = results['nf_pred'][sample_idx, 0, :, :]
        pred_imag = results['nf_pred'][sample_idx, 1, :, :]
        
        # convert to magnitude and phase
        truth_mag = np.sqrt(truth_real**2 + truth_imag**2)
        truth_phase = np.arctan2(truth_imag, truth_real)
        pred_mag = np.sqrt(pred_real**2 + pred_imag**2)
        pred_phase = np.arctan2(pred_imag, pred_real)
        # (if we wanted to keep and plot real/imag)
        '''truth_mag = truth_real
        truth_phase = truth_imag
        pred_mag = pred_real
        pred_phase = pred_imag'''
        
        # 4 subplots (2x2 grid)
        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle(title, fontsize=16)
        fig.text(0.5, 0.92, model_identifier, ha='center', fontsize=12)

        # real part of the truth
        ax[0, 0].imshow(truth_mag, cmap='viridis')
        ax[0, 0].set_title('Truth Magnitude')
        ax[0, 0].axis('off')

        # real part of the prediction
        ax[0, 1].imshow(pred_mag, cmap='viridis')
        ax[0, 1].set_title('Predicted Magnitude')
        ax[0, 1].axis('off')

        # imaginary part of the truth
        ax[1, 0].imshow(truth_phase, cmap='twilight_shifted')
        ax[1, 0].set_title('True Phase')
        ax[1, 0].axis('off')

        # imaginary part of the prediction
        ax[1, 1].imshow(pred_phase, cmap='twilight_shifted')  
        ax[1, 1].set_title('Predicted Phase')
        ax[1, 1].axis('off')

        fig.tight_layout()

        # save the plot if specified
        if save_fig:
            if save_path is None:
                save_path = os.getcwd()
            other_plots_dir = os.path.join(save_path, "other_plots")
            os.makedirs(other_plots_dir, exist_ok=True)
            file_name = f'{title}_dft_sample_idx_{sample_idx}.pdf'
            fig.savefig(os.path.join(other_plots_dir, file_name))
            print(f"Figure saved to {os.path.join(other_plots_dir, file_name)}")

        plt.show()

    # Determine which fold to plot based on plot_type
    if plot_type == "best":
        # Select the fold with the best validation loss
        best_fold_idx = min(range(len(fold_results)), key=lambda i: fold_results[i]['losses']['val_loss'].iloc[-1])
        selected_results = fold_results[best_fold_idx]
        title = f"Best Performing Fold - Fold {best_fold_idx + 1}"
    elif plot_type == "worst":
        # Select the fold with the worst validation loss
        worst_fold_idx = max(range(len(fold_results)), key=lambda i: fold_results[i]['losses']['val_loss'].iloc[-1])
        selected_results = fold_results[worst_fold_idx]
        title = f"Worst Performing Fold - Fold {worst_fold_idx + 1}"
    elif plot_type == "specific" and fold_idx is not None:
        # Plot a specific fold by index
        selected_results = fold_results[fold_idx]
        title = f"Specific Fold -  Fold {fold_idx + 1}"
    else:
        raise ValueError("Invalid plot_type or fold_idx provided")

    # Plot both training and validation results for the selected fold
    plot_single_set(selected_results['train'], f"{title} - Training Dataset", save_dir, sample_idx)
    plot_single_set(selected_results['valid'], f"{title} - Validation Dataset", save_dir, sample_idx)