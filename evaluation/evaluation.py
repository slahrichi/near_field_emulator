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
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

sys.path.append('../')
import utils.mapping as mapping
import utils.visualize as viz

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
    file_path = os.path.join(folder_path, "params.yaml")
    if os.path.isfile(file_path):

        with open(file_path, 'r') as file:
            yaml_content = yaml.safe_load(file)
            key_dict = {}    
            key_dict['arch'] = yaml_content['arch']
            key_dict['lr'] = yaml_content['learning_rate']
            key_dict['optimizer'] = yaml_content['optimizer']
            key_dict['lr_scheduler'] = yaml_content['lr_scheduler']
            key_dict['batch_size'] = yaml_content['batch_size']
            key_dict['loss_func'] = yaml_content['objective_function']
            if yaml_content['arch'] == 0:
                key_dict['mlp_real'] = yaml_content['mlp_real']
                key_dict['mlp_imag'] = yaml_content['mlp_imag']
                if yaml_content['mlp_strategy'] != 0:
                    key_dict['patch_size'] = yaml_content['patch_size']
            elif yaml_content['arch'] == 1:
                key_dict['lstm_num_layers'] = yaml_content['lstm']['num_layers']
                key_dict['lstm_i_dims'] = yaml_content['lstm']['i_dims']
                key_dict['lstm_h_dims'] = yaml_content['lstm']['h_dims']
                key_dict['seq_len'] = yaml_content['seq_len']
            elif yaml_content['arch'] == 2:
                key_dict['in_channels'] = yaml_content['conv_lstm']['in_channels']
                key_dict['out_channels'] = yaml_content['conv_lstm']['out_channels']
                key_dict['kernel_size'] = yaml_content['conv_lstm']['kernel_size']
                key_dict['padding'] = yaml_content['conv_lstm']['padding']
                key_dict['spatial'] = yaml_content['conv_lstm']['spatial']
            key_dict['n_folds'] = yaml_content['n_folds']
            key_dict['title'] = folder_path.replace(excess, "")

            return key_dict
        
def save_eval_item(save_dir, eval_item, file_name, type):
    """Save metrics or plot(s) to a specified file"""
    if 'metrics' in type:
        save_path = os.path.join(save_dir, "performance_metrics")
    elif type == 'loss':
        save_path = os.path.join(save_dir, "loss_plots")
    elif type == 'dft':
        save_path = os.path.join(save_dir, "dft_plots")
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, file_name)
    if 'metrics' in type:
        with open(save_path, 'w') as file:
            for metric, value in eval_item.items():
                if type == 'all_metrics':
                    file.write(f"{metric}: {np.mean(value):.4f} ± {np.std(value):.4f}\n")
                else:
                    file.write(f"{metric}: {value:.4f}\n")
    else:
        eval_item.savefig(save_path)
    print(f"Evaluation item saved to {save_path}")
    

def plot_loss(pm, fold_results, min_list=[None, None], max_list=[None, None], save_fig=False, save_dir=None):
    
    losses = [fold['losses'] for fold in fold_results]
    title = pm.model_id
    lr = pm.learning_rate
    optimizer = pm.optimizer
    lr_scheduler = pm.lr_scheduler
    batch_size = pm.batch_size
    if pm.arch == 0:
        mlp_layers = pm.mlp_real['layers']
        model_identifier = f'{title} - lr: {lr}, lr_scheduler: {lr_scheduler}, optimizer: {optimizer}, batch: {batch_size}, mlp_layers: {mlp_layers}'
        #if params['mlp_strategy'] != 0:
        #    patch_size = params['patch_size']
        #    model_identifier += f", patch_size: {patch_size}"
    elif pm.arch == 1:
        lstm_num_layers = pm.lstm['num_layers']
        lstm_i_dims = pm.lstm['i_dims']
        lstm_h_dims = pm.lstm['h_dims']
        seq_len = pm.seq_len
        model_identifier = f'{title} - lr: {lr}, lr_scheduler: {lr_scheduler}, optimizer: {optimizer}, batch: {batch_size}, lstm_layers: {lstm_num_layers}, i_dims: {lstm_i_dims}, h_dims: {lstm_h_dims}, seq_len: {seq_len}'
    elif pm.arch == 2:
        in_channels = pm.conv_lstm['in_channels']
        out_channels = pm.conv_lstm['out_channels']
        kernel_size = pm.conv_lstm['kernel_size']
        padding = pm.conv_lstm['padding']
        model_identifier = f'{title} - lr: {lr}, lr_scheduler: {lr_scheduler}, optimizer: {optimizer}, batch: {batch_size}, in_channels: {in_channels}, out_channels: {out_channels}, kernel_size: {kernel_size}, padding: {padding}'
    
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
    ax[0].set_ylabel(f"{pm.objective_function} Loss", fontsize=10)
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
    ax[1].set_ylabel(f"{pm.objective_function} Loss", fontsize=10)
    ax[1].set_xlabel("Epoch", fontsize=10)
    ax[1].set_title(f"Validation Loss", fontsize=12)
    ax[1].set_ylim([min_list[1], max_list[1]])
    ax[1].legend()
    
    fig.suptitle(model_identifier)
    fig.tight_layout()
    
    # Save the plot if requested
    if save_fig:
        if not save_dir:
            raise ValueError("Please specify a save directory")
        file_name = f'loss.pdf'
        save_eval_item(save_dir, fig, file_name, 'loss')

def calculate_metrics(truth, pred):
    """Calculate various metrics between ground truth and predictions."""
    mae = np.mean(np.abs(truth - pred))
    rmse = np.sqrt(np.mean((truth - pred) ** 2))
    correlation = np.corrcoef(truth.flatten(), pred.flatten())[0, 1]
    psnr = PeakSignalNoiseRatio(data_range=1.0)(torch.tensor(pred), torch.tensor(truth))
    try: # if spatial size is too small set SSIM to 0
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)(torch.tensor(pred), torch.tensor(truth))
    except:
        ssim = torch.tensor(0.0)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'Correlation': correlation,
        'PSNR': psnr.item(),
        'SSIM': ssim.item()
    }

def print_metrics(fold_results, fold_idx=None, dataset='valid', save_fig=False, save_dir=None):
    """Print metrics for a specific fold and dataset (train or valid)."""
    if fold_idx is not None:
        results = fold_results[fold_idx][dataset]
        truth = results['nf_truth']
        pred = results['nf_pred']
        
        metrics = calculate_metrics(truth, pred)
        print(f"Metrics for Fold {fold_idx + 1} - {dataset.capitalize()} Dataset:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        # save to file
        if save_fig:
            if not save_dir:
                raise ValueError("Please specify a save directory")
            file_name = f'{dataset}_metrics_fold{fold_idx+1}.txt'
            save_eval_item(save_dir, metrics, file_name, 'metrics')
    else:
        # calculate metrics for all folds
        all_metrics = {metric: [] for metric in ['MAE', 'RMSE', 'Correlation', 'PSNR', 'SSIM']}
        for fold_idx, fold in enumerate(fold_results):
            results = fold[dataset]
            truth = results['nf_truth']
            pred = results['nf_pred']
            
            metrics = calculate_metrics(truth, pred)
            for metric, value in metrics.items():
                all_metrics[metric].append(value)
        
        print(f"Average Metrics for All Folds - {dataset.capitalize()} Dataset:")
        for metric, values in all_metrics.items():
            print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
            
        # save to file
        if save_fig:
            if not save_dir:
                raise ValueError("Please specify a save directory")
            file_name = f'{dataset}_metrics.txt'
            save_eval_item(save_dir, all_metrics, file_name, 'all_metrics')

def plot_dft_fields(fold_results, fold_idx=None, plot_type="best", 
                    sample_idx=0, save_fig=False, save_dir=None,
                    arch='mlp', format='polar'):
    """
    Parameters:
    - fold_results: List of dictionaries containing train and valid results for each fold
    - fold_idx: Optional, if you want to visualize a specific fold by index
    - plot_type: "best" to plot best-performing fold, "worst" to plot worst-performing fold,  
      or "specific" if fold_idx is provided  
    - sample_idx: Index of the sample to plot
    - save_fig: Whether to save the plot to a file
    - save_dir: Directory to save the plot to
    - arch: "mlp" or "lstm"
    - format: "cartesian" or "polar"
    """
    def plot_single_set(results, title, format, save_path, sample_idx):
        if arch == 'mlp' or arch == 'autoencoder':
            # extract and convert to tensors
            truth_real = torch.from_numpy(results['nf_truth'][sample_idx, 0, :, :])
            truth_imag = torch.from_numpy(results['nf_truth'][sample_idx, 1, :, :])
            pred_real = torch.from_numpy(results['nf_pred'][sample_idx, 0, :, :])
            pred_imag = torch.from_numpy(results['nf_pred'][sample_idx, 1, :, :])

            # determine which coordinate format to plot
            if format == 'polar':
                component_1 = "Magnitude"
                component_2 = "Phase"
                # convert to magnitude and phase
                truth_component_1, truth_component_2 = mapping.cartesian_to_polar(truth_real, truth_imag)
                pred_component_1, pred_component_2 = mapping.cartesian_to_polar(pred_real, pred_imag)
            else:
                component_1 = "Real"
                component_2 = "Imaginary"
                truth_component_1, truth_component_2 = truth_real, truth_imag
                pred_component_1, pred_component_2 = pred_real, pred_imag

            # 4 subplots (2x2 grid)
            fig, ax = plt.subplots(2, 2, figsize=(8, 8))
            fig.suptitle(title, fontsize=16)
            fig.text(0.5, 0.92, model_identifier, ha='center', fontsize=12)

            # real part of the truth
            ax[0, 0].imshow(truth_component_1, cmap='viridis')
            ax[0, 0].set_title(f'True {component_1} Component')
            ax[0, 0].axis('off')

            # real part of the prediction
            ax[0, 1].imshow(pred_component_1, cmap='viridis')
            ax[0, 1].set_title(f'Predicted {component_1} Component')
            ax[0, 1].axis('off')

            # imaginary part of the truth
            ax[1, 0].imshow(truth_component_2, cmap='twilight_shifted')
            ax[1, 0].set_title(f'True {component_2} Component')
            ax[1, 0].axis('off')

            # imaginary part of the prediction
            ax[1, 1].imshow(pred_component_2, cmap='twilight_shifted')  
            ax[1, 1].set_title(f'Predicted {component_2} Component')
            ax[1, 1].axis('off')
                
        elif arch == 'lstm' or arch == 'convlstm':
            # extract and convert to tensors
            truth_real = torch.from_numpy(results['nf_truth'][sample_idx, :, 0, :, :])
            truth_imag = torch.from_numpy(results['nf_truth'][sample_idx, :, 1, :, :])
            pred_real = torch.from_numpy(results['nf_pred'][sample_idx, :, 0, :, :])
            pred_imag = torch.from_numpy(results['nf_pred'][sample_idx, :, 1, :, :])
            
            # determine which coordinate format to plot
            if format == 'polar':
                component_1 = "Magnitude"
                component_2 = "Phase"
                # convert to magnitude and phase
                truth_component_1, truth_component_2 = mapping.cartesian_to_polar(truth_real, truth_imag)
                pred_component_1, pred_component_2 = mapping.cartesian_to_polar(pred_real, pred_imag)
            else:
                component_1 = "Real"
                component_2 = "Imaginary"
                truth_component_1, truth_component_2 = truth_real, truth_imag
                pred_component_1, pred_component_2 = pred_real, pred_imag
            
            seq_len = truth_component_1.shape[0]
            
            # Create figure WITHOUT creating subplots
            fig = plt.figure(figsize=(4*seq_len + 2, 16))
            
            # Create gridspec with space for labels and column headers
            gs = fig.add_gridspec(5, seq_len + 1,  # 5 rows: header + 4 data rows
                                width_ratios=[0.3] + [1]*seq_len,
                                height_ratios=[0.05] + [1]*4,
                                hspace=0.1,
                                wspace=0.1)
            
            # Create axes for column headers
            header_axs = [fig.add_subplot(gs[0, j]) for j in range(1, seq_len + 1)]
            
            # Create axes for images
            axs = [[fig.add_subplot(gs[i+1, j]) for j in range(1, seq_len + 1)] 
                for i in range(4)]
            
            # Create axes for row labels
            label_axs = [fig.add_subplot(gs[i+1, 0]) for i in range(4)]
            
            fig.suptitle(title, fontsize=24, y=0.95, fontweight='bold')
            fig.text(0.5, 0.94, model_identifier, ha='center', fontsize=16)
            
            # Add column headers
            for t, ax in enumerate(header_axs):
                ax.axis('off')
                ax.text(0.5, 0.3,
                    f't={t+1}',
                    ha='center',
                    va='center',
                    fontsize=20,
                    fontweight='bold')
            
            # Add row labels
            row_labels = [f'Ground Truth\n{component_1}',
                        f'Predicted\n{component_1}',
                        f'Ground Truth\n{component_2}',
                        f'Predicted\n{component_2}']
            
            for ax, label in zip(label_axs, row_labels):
                ax.axis('off')
                ax.text(0.95, 0.5, 
                    label,
                    ha='right',
                    va='center',
                    fontsize=20,
                    fontweight='bold')
            
            # Plot sequence
            for t in range(seq_len):
                axs[0][t].imshow(truth_component_1[t], cmap='viridis')
                axs[0][t].axis('off')
                
                axs[1][t].imshow(pred_component_1[t], cmap='viridis')
                axs[1][t].axis('off')
                
                axs[2][t].imshow(truth_component_2[t], cmap='twilight_shifted')
                axs[2][t].axis('off')
                
                axs[3][t].imshow(pred_component_2[t], cmap='twilight_shifted')
                axs[3][t].axis('off')
                
        fig.tight_layout()

        # save the plot if specified
        if save_fig:
            if not save_dir:
                raise ValueError("Please specify a save directory")
            file_name = f'{title}_dft_sample_idx_{sample_idx}_{format}.pdf'
            save_eval_item(save_dir, fig, file_name, 'dft')

        plt.show()

    # Determine which fold to plot based on plot_type
    if plot_type == "best":
        # Select the fold with the best validation loss
        best_fold_idx = min(range(len(fold_results)), key=lambda i: fold_results[i]['losses']['val_loss'].iloc[-2])
        selected_results = fold_results[best_fold_idx]
        title = f"Best Performing Fold - Fold {best_fold_idx + 1}"
    elif plot_type == "worst":
        # Select the fold with the worst validation loss
        worst_fold_idx = max(range(len(fold_results)), key=lambda i: fold_results[i]['losses']['val_loss'].iloc[-2])
        selected_results = fold_results[worst_fold_idx]
        title = f"Worst Performing Fold - Fold {worst_fold_idx + 1}"
    elif plot_type == "specific" and fold_idx is not None:
        # Plot a specific fold by index
        selected_results = fold_results[fold_idx]
        title = f"Specific Fold -  Fold {fold_idx + 1}"
    else:
        raise ValueError("Invalid plot_type or fold_idx provided")

    # Plot both training and validation results for the selected fold
    plot_single_set(selected_results['train'], f"{title} - Random Training Sample - {format}", format, save_dir, sample_idx)
    plot_single_set(selected_results['valid'], f"{title} - Random Validation Sample - {format}", format, save_dir, sample_idx)
    
def plot_sequence_comparison(pred, truth, view='mag', save_fig=False, save_dir=None, title=None):
    """
    Plot a sequence of predicted vs ground truth fields for either magnitude or phase
    
    Args:
        pred (tensor): Predicted fields of shape [2, 166, 166, 5]
        truth (tensor): Ground truth fields of shape [2, 166, 166, 5]
        view (str): Either 'mag' or 'phase' to specify which component to view
        save_fig (bool): Whether to save the figure
        save_dir (str): Directory to save figure if save_fig is True
        title (str): Optional title for the plot
    """
    # Convert from real/imaginary to magnitude/phase
    pred_real = pred[0]  # [166, 166, 5]
    pred_imag = pred[1]
    truth_real = truth[0]
    truth_imag = truth[1]
    
    pred_mag, pred_phase = mapping.cartesian_to_polar(pred_real, pred_imag)
    truth_mag, truth_phase = mapping.cartesian_to_polar(truth_real, truth_imag)
    
    # Select which view to display
    if view.lower() == 'mag':
        pred_view = pred_mag
        truth_view = truth_mag
        cmap = 'viridis'
        component = 'Magnitude'
    elif view.lower() == 'phase':
        pred_view = pred_phase
        truth_view = truth_phase
        cmap = 'twilight_shifted'
        component = 'Phase'
    else:
        raise ValueError("view must be either 'mag' or 'phase'")
    
    seq_len = pred_view.shape[-1]
    
    # create figure with 2 rows (truth/pred) and seq_len columns
    fig, axs = plt.subplots(2, seq_len, figsize=(4*seq_len, 8))
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    # plot each timestep
    for t in range(seq_len):
        # truth
        axs[0, t].imshow(truth_view[..., t], cmap=cmap)
        axs[0, t].set_title(f'Ground Truth {component}\nt={t+1}')
        axs[0, t].axis('off')
        
        # prediction
        axs[1, t].imshow(pred_view[..., t], cmap=cmap)
        axs[1, t].set_title(f'Predicted {component}\nt={t+1}')
        axs[1, t].axis('off')
    
    plt.tight_layout()
    
    # save if requested
    if save_fig:
        if not save_dir:
            raise ValueError("Please specify a save directory")
        file_name = f'sequence_comparison_{view}_{title}.pdf' if title else f'sequence_comparison_{view}.pdf'
        save_eval_item(save_dir, fig, file_name, 'dft')
    
    plt.show()
    
def animate_fields(fold_results, dataset, fold_idx=0, sample_idx=0, seq_len=5, save_dir=None): 
    results = fold_results[fold_idx][dataset]
    truth_real = torch.from_numpy(results['nf_truth'][sample_idx, :, 0, :, :])
    truth_imag = torch.from_numpy(results['nf_truth'][sample_idx, :, 1, :, :])
    pred_real = torch.from_numpy(results['nf_pred'][sample_idx, :, 0, :, :])
    pred_imag = torch.from_numpy(results['nf_pred'][sample_idx, :, 1, :, :])
    truth_real = truth_real.permute(1, 2, 0)
    truth_imag = truth_imag.permute(1, 2, 0)
    pred_real = pred_real.permute(1, 2, 0)
    pred_imag = pred_imag.permute(1, 2, 0)

    truth_mag, truth_phase = mapping.cartesian_to_polar(truth_real, truth_imag)
    pred_mag, pred_phase = mapping.cartesian_to_polar(pred_real, pred_imag)

    flipbooks_dir = os.path.join(save_dir, "flipbooks")
    os.makedirs(flipbooks_dir, exist_ok=True)

    # intensity
    truth_anim = viz.animate_fields(truth_mag, "True Intensity", 
                                    save_path=os.path.join(flipbooks_dir, f"sample_{sample_idx}_mag_groundtruth_{dataset}.gif"), 
                                    frames=seq_len,
                                    interval=250)
    pred_anim = viz.animate_fields(pred_mag, "Predicted Intensity", 
                                save_path=os.path.join(flipbooks_dir, f"sample_{sample_idx}_mag_prediction_{dataset}.gif"), 
                                frames=seq_len,
                                interval=250)

    # phase
    truth_phase_anim = viz.animate_fields(truth_phase, "True Phase", 
                                    save_path=os.path.join(flipbooks_dir, f"sample_{sample_idx}_phase_groundtruth_{dataset}.gif"), 
                                    cmap='twilight_shifted',
                                    frames=seq_len,
                                    interval=250)
    pred_phase_anim = viz.animate_fields(pred_phase, "Predicted Phase", 
                                save_path=os.path.join(flipbooks_dir, f"sample_{sample_idx}_phase_prediction_{dataset}.gif"), 
                                cmap='twilight_shifted',
                                frames=seq_len,
                                interval=250)