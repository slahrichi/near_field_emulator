#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import os
import yaml
import torch
import logging
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

from utils.mapping import get_model_type
from core import datamodule
from core.autoencoder import Autoencoder
from core.WaveMLP import WaveMLP
import core.WaveModel as models

def select_model(pm, fold_idx=None):
    logging.debug("select_model.py - Selecting model") 
    model_type = get_model_type(pm.arch)
    if model_type == 'autoencoder': # autoencoder pretraining
        network = Autoencoder(pm.params_model, fold_idx)
    elif model_type == 'mlp' or model_type == 'cvnn':
        pm.params_model['name'] = model_type
        network = WaveMLP(pm.params_model, fold_idx)
    # mode lstm is just the lstm but on epre-encoded data
    elif model_type == 'lstm' or model_type == 'modelstm':
        pm.params_model['name'] = model_type
        network = models.WaveLSTM(pm.params_model, fold_idx)
    elif model_type == 'convlstm':
        pm.params_model['name'] = model_type
        network = models.WaveConvLSTM(pm.params_model, fold_idx)
    elif model_type == 'ae-lstm':
        pm.params_model['name'] = model_type
        network = models.WaveAELSTM(pm.params_model, fold_idx)
    elif model_type == 'ae-convlstm':
        pm.params_model['name'] = model_type
        network = models.WaveAEConvLSTM(pm.params_model, fold_idx)
    else:
        raise NotImplementedError("Model type not recognized.")

    if pm.load_checkpoint:
         
        checkpoint_path = os.path.join(pm.path_root, pm.path_checkpoint)
        checkpoint = os.listdir(checkpoint_path)[0]
        checkpoint = os.path.join(checkpoint_path, checkpoint)
        print(checkpoint)
        
        state_dict = torch.load(checkpoint)['state_dict']
        #network.load_from_checkpoint(pm.path_checkpoint,
        #                                   params = (pm.params_model, pm.params_propagator),
        #                                   strict = True)
        network.load_state_dict(state_dict, strict=True)

    assert network is not None

    return network
