#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import sys
import torch
import numpy as np
#from geomloss import SamplesLoss
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
#from torchvision.models import resnet50, resnet18, resnet34
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from pytorch_lightning import LightningModule
import math
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
#from utils import parameter_manager
#from core.complexNN import ComplexLinear, ModReLU
from .CVNN import ComplexReLU, ModReLU, ComplexLinearFinal

from .WaveMLP import WaveMLP
from conf.schema import load_config

sys.path.append("../")

class WaveInverseMLP(LightningModule):
    """Radii Prediction Model  
    Architecture: MLPs  
    Modes: Simple Inverse, Tandem"""
    def __init__(self, model_config, fold_idx=None):
        super().__init__()
        
        self.conf = model_config
        self.learning_rate = self.conf.learning_rate
        self.lr_scheduler = self.conf.lr_scheduler
        self.loss_func = self.conf.objective_function
        self.fold_idx = fold_idx
        self.patch_size = self.conf.patch_size # for non-default approaches
        self.name = self.conf.arch # inverse
        self.num_design_conf = int(self.conf.num_design_conf)
        self.strat = None
        self.forward_ckpt_path = self.conf.forward_ckpt_path
        self.forward_config_path = self.conf.forward_config_path

        
        # inverse (geometry --> radii)
        # for now, we will just load a trained forward model
        # we are also using CVNN
        self.output_size = 9
        if self.name == "inverse":
            self.cvnn = self.build_mlp(self.num_design_conf, self.conf.cvnn)
        else:
            raise ValueError("Inverse model not supported with multiple MLPs yet")


        self.save_hyperparameters()
        
        # Store necessary lists for tracking metrics per fold
        self.test_results = {'train': {'radii_pred': [], 'radii_truth': [], 'field_resim': [], 'field_truth': []},
                            'valid': {'radii_pred': [], 'radii_truth': [],  'field_resim': [], 'field_truth': []}}
         
    def build_mlp(self, input_size, mlp_conf):
        layers = []
        in_features = input_size
        for layer_size in mlp_conf['layers']:
            if self.name == 'inverse': # complex-valued NN
                layers.append(ComplexLinear(in_features, layer_size))
            else: # real-valued NN
                raise ValueError(f"Only supporting CVNN for inverse for now. You are trying to use {self.name}")
            layers.append(self.get_activation_function(mlp_conf['activation']))
            in_features = layer_size
        if self.name == 'inverse':
            # this should actually be self.name == "cvnn",
            # TODO: in the future, if we want to use non-complex NN, we would need to add a flag to differentiate CVNN from MLP in the Inverse Setting. 
            layers.append(ComplexLinearFinal(in_features, self.output_size))
        else:
            raise ValueError(f"Only supporting CVNN for inverse for now. You are trying to use {self.name}")
        return nn.Sequential(*layers)
        
    def get_activation_function(self, activation_name):
        if activation_name == 'relu':
            return nn.ReLU()
        elif activation_name == 'sigmoid':
            return nn.Sigmoid()
        elif activation_name == 'tanh':
            return nn.Tanh()
        elif activation_name == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation_name == 'modrelu':
            return ModReLU()
        elif activation_name == 'complexrelu':
            return ComplexReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")
        
    def forward(self, input):
        # Inverse model: near_fields -> radii
        near_fields = input
        near_fields_complex = torch.complex(near_fields[:, 0, :, :], near_fields[:, 1, :, :])
        near_fields_flat = near_fields_complex.view(near_fields_complex.size(0), -1)
        output = self.cvnn(near_fields_flat)
        output.view(-1, 9)
        return output
  
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # setup specified scheduler
        if self.lr_scheduler == 'ReduceLROnPlateau':
            choice = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                    factor=0.5, patience=5, 
                                                                    min_lr=1e-6, threshold=0.001, 
                                                                    cooldown=2)
        elif self.lr_scheduler == 'CosineAnnealingLR':
            choice = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                T_max=100,
                                                                eta_min=1e-6)
        elif self.lr_scheduler == 'None':
            return optimizer
        else:
            raise ValueError(f"Unsupported LR scheduler: {self.lr_scheduler}")
        
        scheduler = {
            'scheduler': choice,
            'interval': 'epoch',
            'monitor': 'val_loss',
            'frequency': 1
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def compute_loss(self, near_fields, preds, labels, choice):
        # Inverse: preds/labels are radii
        if choice == 'mse':
            # Mean Squared Error for simple inverse
            preds = preds.to(torch.float32).contiguous()
            labels = labels.to(torch.float32).contiguous()
            fn = torch.nn.MSELoss()
            loss = fn(preds, labels)
        elif choice == "resim":
            # Resimulation Error for Tandem (using trained forward model)
            preds = preds.to(torch.float32).contiguous()
            labels = labels.to(torch.float32).contiguous()
            mse = torch.nn.MSELoss()
            
            fwd = self.load_forward(self.forward_ckpt_path, self.forward_config_path)
            fwd_pred = fwd(preds)
            
            fwd_pred_real = fwd_pred.real
            fwd_pred_imag = fwd_pred.imag

            near_fields_complex = torch.complex(near_fields[:, 0, :, :], near_fields[:, 1, :, :])
            near_fields_real = near_fields_complex.real
            near_fields_imag = near_fields_complex.imag

            loss_real = mse(fwd_pred_real, near_fields_real)
            loss_imag = mse(fwd_pred_imag, near_fields_imag)
            loss = loss_real + loss_imag
        else:
            raise ValueError(f"Unsupported loss function: {choice}")
            
        return loss
    
    def load_forward(self, checkpoint_path, config_path):
        forward_conf = load_config(config_path)
        model = WaveMLP(model_config=forward_conf.model)
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        model.to(self.device)
        model.eval()

        return model

    def objective(self, batch, predictions):
        near_fields, radii = batch
        if self.name == 'inverse':
            labels = radii  
            preds = predictions.real
            radii_loss = self.compute_loss(near_fields, preds, labels, choice=self.loss_func)
        else:
            raise ValueError("Only CVNN handled for inverse!")
        
            # compute other metrics for logging besides specified loss function
        choices = {
            'mse': None,
            'resim': None
        }
        for key in choices:
            if key != self.loss_func:
                loss = self.compute_loss(near_fields, preds, labels, choice=key)
                choices[key] = loss
        return {"loss": radii_loss, **choices}


    def shared_step(self, batch, batch_idx):
        near_fields, radii = batch
        preds = self.forward(near_fields)
        return preds
    
    def training_step(self, batch, batch_idx):
        preds = self.shared_step(batch, batch_idx)
        loss_dict = self.objective(batch, preds)
        loss = loss_dict['loss']
        
        # log metrics
        if self.loss_func == 'psnr' or self.loss_func == 'ssim': # PSNR is inverted for minimization, report true value here
            self.log('train_loss', -loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        else: # mse
            self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        other_metrics = [f"{key}" for key in loss_dict.keys() if key != 'loss' and key != self.loss_func]
        for key in other_metrics:
            #print(f"train_{key}", loss_dict[key])
            self.log(f"train_{key}", loss_dict[key], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'loss': loss, 'output': preds, 'target': batch}
    
    def validation_step(self, batch, batch_idx):
        preds = self.shared_step(batch, batch_idx)
        loss_dict = self.objective(batch, preds)
        loss = loss_dict['loss']
        
        # log metrics
        if self.loss_func == 'psnr' or self.loss_func == 'ssim': # PSNR is inverted for minimization, report true value here
            self.log('val_loss', -loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        else: # mse
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        #print(f"val_psnr_recorded: {-loss}")
        other_metrics = [f"{key}" for key in loss_dict.keys() if key != 'loss' and key != self.loss_func]
        for key in other_metrics:
            #print(f"valid_{key}", loss_dict[key])
            self.log(f"valid_{key}", loss_dict[key], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'loss': loss, 'output': preds, 'target': batch}
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        preds = self.shared_step(batch, batch_idx)   
        self.organize_testing(preds, batch, batch_idx, dataloader_idx)
        
    def organize_testing(self, predictions, batch, batch_idx, dataloader_idx):
        near_fields, radii = batch
        if self.name == 'inverse':
            # Saving real part of the prediction only; double check that imaginary part can be disregarded
            preds_real = predictions.real

            # Compute resimulated fields
            fwd = self.load_forward(self.forward_ckpt_path, self.forward_config_path)
            field_resim = fwd(preds_real) # check if CVNN takes real only
            field_truth = torch.complex(near_fields[:, 0, :, :], near_fields[:, 1, :, :])
            field_real = field_truth.real
            field_imag = field_truth.imag
            field_resim_real = field_resim.real
            field_resim_imag = field_resim.imag
            resim_combined = torch.stack([field_resim_real, field_resim_imag], dim=1).cpu().numpy()
            field_combined = torch.stack([field_real, field_imag], dim=1).cpu().numpy()

            # Store predictions and ground truths for analysis after testing
            if dataloader_idx == 0:  # val dataloader
                self.test_results['valid']['radii_pred'].append(preds_real)
                self.test_results['valid']['radii_truth'].append(radii)
                self.test_results['valid']['field_resim'].append(resim_combined)
                self.test_results['valid']['field_truth'].append(field_combined)
            elif dataloader_idx == 1:  # train dataloader
                self.test_results['train']['radii_pred'].append(preds_real)
                self.test_results['train']['radii_truth'].append(radii)
                self.test_results['train']['field_resim'].append(resim_combined)
                self.test_results['train']['field_truth'].append(field_combined)
            else:
                raise ValueError(f"Invalid dataloader index: {dataloader_idx}")
        else:
            raise ValueError("Testing for inverse only available for CVNN.")

    def on_test_end(self):
        # Concatenate results from all batches
        for mode in ['train', 'valid']:
            out = 'radii'
            self.test_results[mode][f'{out}_pred'] = np.concatenate([tensor.cpu().detach().numpy() for tensor in self.test_results[mode][f'{out}_pred']], axis=0)
            self.test_results[mode][f'{out}_truth'] = np.concatenate([tensor.cpu().detach().numpy() for tensor in self.test_results[mode][f'{out}_truth']], axis=0)
            # Log results for the current fold
            '''name = "results"
            self.logger.experiment.log_results(
                results=self.test_results[mode],
                epoch=None,
                mode=mode,
                name=name
            )'''