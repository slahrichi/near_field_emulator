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

from core import datamodule, model


def select_model(pm, fold_idx=None):
    logging.debug("select_model.py - Selecting model") 
    if pm.arch == 0:
        network = model.WaveMLP(pm.params_model, fold_idx)
    elif pm.which == 1:
        network = model.WaveLSTM(pm.params_model, fold_idx)
    elif pm.which == 2:
        # TODO ConvLSTM
        raise NotImplementedError
    else:
        raise ValueError("Model not recognized")

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
