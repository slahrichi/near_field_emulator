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
from utils import parameter_manager, mapping
from core import datamodule as dm

def run(conf):
    """This function finishes the preprocessing pipeline for the data,  
       fully loading it into the PyTorch format expected by the dataloader.  
       It operates on the preprocessed pickle files separated by train/valid.  
       in the preprocessed_data directory.
    """
    #logging.basicConfig(level=logging.DEBUG)
    seed_everything(1337)
    logging.basicConfig(level=logging.DEBUG)

    # for accessing the preprocessed data
    train_path = os.path.join(conf.paths.data, 'train')
    valid_path = os.path.join(conf.paths.data, 'valid')
    
    model_type = conf.model.arch
    
    # Load data
    if model_type == 'mlp' or model_type == 'cvnn':
        save_path = os.path.join(conf.paths.data, 'dataset_nobuffer.pt')
        if os.path.exists(save_path):
            raise FileExistsError(f"Output file {save_path} already exists!")
        dm.load_pickle_data(train_path, valid_path, save_path, arch='mlp')
    else: # LSTM
        save_path = os.path.join(conf.paths.data, 'dataset.pt')
        logging.debug(f"Save path: {save_path}")
        logging.debug(f"Save directory exists: {os.path.exists(os.path.dirname(save_path))}")
        logging.debug(f"Save directory writable: {os.access(os.path.dirname(save_path), os.W_OK)}")
        #if os.path.exists(save_path):
        #    raise FileExistsError(f"Output file {save_path} already exists!")
        dm.load_pickle_data(train_path, valid_path, save_path, arch='lstm')