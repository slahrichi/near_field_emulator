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

def load_loss(path):
    if is_csv_empty(path):
        print("Empty CSV file")
        return False
    else:
        return pd.read_csv(path)


def get_results(res_path, folder_name):

    train_path = os.path.join(res_path, folder_name, "train_info")
    valid_path = os.path.join(res_path, folder_name, "valid_info")

    loss_file = os.path.join(res_path, folder_name, "loss.csv")
    loss = load_loss(loss_file)

    train_results = pickle.load(open(os.path.join(train_path, "results.pkl"), "rb"))
    valid_results = pickle.load(open(os.path.join(valid_path, "results.pkl"), "rb"))
    
    return loss, train_results, valid_results

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
            key_dict['title'] = folder_path.replace(excess, "")

            loss_file = os.path.join(folder_path, "loss.csv")
            loss = load_loss(loss_file)
            if type(loss) == bool and loss == False:
                print("Failed to load loss")
                pass
            else:
                key_dict['loss'] = loss
                return key_dict

def plot_loss(model_info, min_list, max_list, save_fig=False, save_dir=None):
    
    loss = model_info['loss']
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
    
    # Assuming there are only two losses now: training and validation
    lterms = ['Near Field Loss']
    
    for i, name in enumerate(loss.keys()):
        if name == "epoch":
            continue
        
        if name.startswith('val'):
            x_vals = loss["epoch"]
            x_vals = x_vals[x_vals.index % 2 == 0]
            y_vals = loss[name]
            y_vals = y_vals[y_vals.index % 2 == 0]
            # Plot validation loss
            ax[1].plot(x_vals, y_vals, color='blue', label=f'{title} - Validation')
            ax[1].set_ylabel("Loss", fontsize=10)
            ax[1].set_xlabel("Epoch", fontsize=10)
            ax[1].set_title(f"Validation Loss", fontsize=12)
            ax[1].set_ylim([min_list[i], max_list[i]])
        else:
            x_vals = loss["epoch"]
            x_vals = x_vals[x_vals.index % 2 != 0]
            y_vals = loss[name]
            y_vals = y_vals[y_vals.index % 2 != 0]
            # Plot training loss
            ax[0].plot(x_vals, y_vals, color='red', label=f'{title} - Training')
            ax[0].set_ylabel("Loss", fontsize=10)
            ax[0].set_xlabel("Epoch", fontsize=10)
            ax[0].set_title(f"Training Loss", fontsize=12)
            ax[0].set_ylim([min_list[i], max_list[i]])

    fig.suptitle(model_identifier)
    fig.tight_layout()

    if save_fig:
        if save_dir is None:
            save_dir = os.getcwd()
        loss_plots_dir = os.path.join(save_dir, "loss_plots")
        os.makedirs(loss_plots_dir, exist_ok=True)
        fig.savefig(os.path.join(loss_plots_dir, f'{title}.pdf'))
        print(f"Figure saved to {os.path.join(loss_plots_dir, f'{title}.pdf')}")
        
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_dft_fields(train_results, valid_results, sample_idx=0, save_fig=False, save_dir=None):
    def plot_single_set(results, title, save_path, sample_idx):
        # extract the specific sample from the results
        truth_real = results['nf_truth'][sample_idx, 0, :, :]
        truth_imag = results['nf_truth'][sample_idx, 1, :, :]
        pred_real = results['nf_pred'][sample_idx, 0, :, :]
        pred_imag = results['nf_pred'][sample_idx, 1, :, :]
        
        # 4 subplots (2x2 grid)
        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle(title, fontsize=16)
        fig.text(0.5, 0.92, model_identifier, ha='center', fontsize=12)

        # real part of the truth
        ax[0, 0].imshow(truth_real, cmap='viridis')
        ax[0, 0].set_title('Truth Intensity')
        ax[0, 0].axis('off')

        # real part of the prediction
        ax[0, 1].imshow(pred_real, cmap='viridis')
        ax[0, 1].set_title('Predicted Intensity')
        ax[0, 1].axis('off')

        # imaginary part of the truth
        ax[1, 0].imshow(truth_imag, cmap='twilight_shifted')
        ax[1, 0].set_title('True Phase')
        ax[1, 0].axis('off')

        # imaginary part of the prediction
        ax[1, 1].imshow(pred_imag, cmap='twilight_shifted')  
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

    plot_single_set(train_results, "Training Dataset", save_dir, sample_idx)
    plot_single_set(valid_results, "Validation Dataset", save_dir, sample_idx)