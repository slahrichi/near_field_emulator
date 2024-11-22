#--------------------------------
# Import: Basic Python Libraries
#--------------------------------
import logging
import yaml
import os
import sys
from pytorch_lightning import seed_everything

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
sys.path.append('../')
from utils import parameter_manager
from core import datamodule as dm

def compile_data(params):
    """This function finishes the preprocessing pipeline for the data,  
       fully loading it into the PyTorch format expected by the dataloader.  
       It operates on the preprocessed pickle files separated by train/valid.  
       in the preprocessed_data directory.
    """
    #logging.basicConfig(level=logging.DEBUG)
    seed_everything(1337)
    logging.basicConfig(level=logging.DEBUG)

    params['model_id'] = 0

    #Parameter manager
    pm = parameter_manager.Parameter_Manager(params=params)

    # for accessing the preprocessed data
    train_path = os.path.join(params['path_root'], params['path_data'], 'train')
    valid_path = os.path.join(params['path_root'], params['path_data'], 'valid')
    
    # Load data
    if pm.arch == 0: # MLP
        save_path = os.path.join(params['path_root'], params['path_data'], 'dataset.pt')
        if os.path.exists(save_path):
            raise FileExistsError(f"Output file {save_path} already exists!")
        dm.load_pickle_data(train_path, valid_path, save_path, arch='mlp')
    elif pm.arch == 1 or params['arch'] == 2: # LSTM
        save_path = os.path.join(params['path_root'], params['path_data'], 'dataset.pt')
        logging.debug(f"Save path: {save_path}")
        logging.debug(f"Save directory exists: {os.path.exists(os.path.dirname(save_path))}")
        logging.debug(f"Save directory writable: {os.access(os.path.dirname(save_path), os.W_OK)}")
        if os.path.exists(save_path):
            raise FileExistsError(f"Output file {save_path} already exists!")
        dm.load_pickle_data(train_path, valid_path, save_path, arch='lstm')
    else:
        logging.error("datamodule.py | Dataset {} not implemented!".format(params['which']))
        exit()