#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import sys
import torch
import numpy as np
import os
#from geomloss import SamplesLoss
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from pytorch_lightning import LightningModule
import math
import time
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

class WaveInverseConvMLP(LightningModule):
    """Radii Prediction Model  
    Architecture: ConvMLPs  
    Modes: Tandem"""
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
        self.radii_bounds = self.conf.radii_bounds
        self.radii_lower_bound = torch.tensor([self.radii_bounds[0]] * 9, device=self.device, dtype=torch.float32)
        self.radii_upper_bound = torch.tensor([self.radii_bounds[1]] * 9, device=self.device, dtype=torch.float32)
        self.register_buffer('radii_mean', (self.radii_lower_bound + self.radii_upper_bound) / 2)
        self.register_buffer('radii_range', self.radii_upper_bound - self.radii_lower_bound)
        #self.radii_range = self.radii_upper_bound - self.radii_lower_bound
        #self.radii_mean = (self.radii_lower_bound + self.radii_upper_bound) / 2
        self.model_id = self.conf.model_id
        self.save_dir = f'/develop/results/meep_meep/{self.name}/model_{self.model_id}'
        self.conv_out_channels = self.conf.conv_out_channels

        
        # inverse (geometry --> radii)
        # for now, we will just load a trained forward model
        # we are also using CVNN
        self.output_size = 9
        self.cvnn = self.build_mlp(self.num_design_conf, self.conf.cvnn)
        
        self.save_hyperparameters()
        
        # Store necessary lists for tracking metrics per fold
        self.test_results = {'train': {'radii_pred': [], 'radii_truth': [], 'field_resim': [], 'field_truth': []},
                            'valid': {'radii_pred': [], 'radii_truth': [],  'field_resim': [], 'field_truth': []}}
         
    def build_mlp(self, input_size, mlp_conf):
        if mlp_conf.get('use_resnet', False):
            # Initialize pre-trained ResNet18
            model = resnet18(pretrained=True)
            
            # Modify first conv layer to accept 2 channels (real and imaginary parts)
            original_layer = model.conv1
            model.conv1 = nn.Conv2d(2, 64, 
                                  kernel_size=7, 
                                  stride=2, 
                                  padding=3, 
                                  bias=False)
            
            # Initialize the weights of the new conv layer
            with torch.no_grad():
                model.conv1.weight[:, :2] = original_layer.weight[:, :2]
            
            # Replace final fully connected layer
            model.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, self.output_size)
            )
            return model
        else:
            # Original ConvMLP architecture
            layers = []
            in_channels = 1

            conv_layers_conf = mlp_conf.get('conv_layers', [])
            for conf in conv_layers_conf:
                out_channels, kernel_size, stride = conf
                layers.append(ComplexConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
                layers.append(self.get_activation_function(mlp_conf['activation']))
                in_channels = out_channels
            layers.append(nn.Flatten())

            dummy_input = torch.randn(1, 1, 166, 166)
            dummy_complex_input = torch.complex(dummy_input, dummy_input)
            with torch.no_grad():
                conv_output_shape = nn.Sequential(*layers[:-1])(dummy_complex_input).shape
            
            in_features = conv_output_shape[1] * conv_output_shape[2] * conv_output_shape[3]

            linear_layers_conf = mlp_conf.get('layers', []) 
            for layer_size in linear_layers_conf:
                layers.append(ComplexLinear(in_features, layer_size))
                layers.append(self.get_activation_function(mlp_conf['activation']))
                in_features = layer_size
            layers.append(ComplexLinearFinal(in_features, self.output_size))

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
        
        if self.conf.cvnn.get('use_resnet', False):
            # For ResNet: use real-valued input with 2 channels
            output = self.cvnn(near_fields)  # near_fields is already [batch_size, 2, 166, 166]
        else:
            # For original ComplexConv architecture
            near_fields_complex = torch.complex(near_fields[:, 0, :, :], near_fields[:, 1, :, :])
            near_fields_complex = near_fields_complex.unsqueeze(1)
            output = self.cvnn(near_fields_complex)
        
        output = output.view(-1, self.output_size)
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
        elif choice == "resim_bdy":
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
            mse_loss = loss_real + loss_imag
            
            relu = torch.nn.ReLU()
            bdy_loss_all = relu(torch.abs(preds - self.radii_mean) - 0.5 * self.radii_range)
            bdy_loss = torch.mean(bdy_loss_all) * 10
            
            loss = torch.add(mse_loss, bdy_loss)
        elif choice == "sensitivity":
            # replacing prediction with ground truth
            preds = preds.to(torch.float32).contiguous()
            labels = labels.to(torch.float32).contiguous()
            mse = torch.nn.MSELoss()
            noise_std = 0.01
            noise = torch.randn_like(near_fields) * noise_std
            fwd_pred = near_fields + noise
            print(fwd_pred - noise)
            loss = mse(fwd_pred, near_fields)
        else:
            raise ValueError(f"Unsupported loss function: {choice}")
            
        return loss
    
    def load_forward(self, checkpoint_path, config_path):
        forward_conf = load_config(config_path)
        model = WaveMLP(model_config=forward_conf.model)
        model.load_state_dict(torch.load(checkpoint_path, weights_only=False)['state_dict'])
        model.to(self.device)
        model.eval()

        return model

    def objective(self, batch, predictions):
        near_fields, radii = batch
        labels = radii  
        preds = predictions.real
        radii_loss = self.compute_loss(near_fields, preds, labels, choice=self.loss_func)
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
        start_time = time.time()
        preds = self.forward(near_fields)
        elapsed_time = time.time() - start_time
        print(f"Time to predict batch {batch_idx}: {elapsed_time:.6f} seconds")
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
            # Convert to tensors and save as lists
            valid_radii_pred = [pred for pred in self.test_results['valid']['radii_pred']]
            valid_radii_truth = [truth for truth in self.test_results['valid']['radii_truth']]
            valid_field_resim = [resim for resim in self.test_results['valid']['field_resim']]
            valid_field_truth = [truth for truth in self.test_results['valid']['field_truth']]

            # Save as lists of tensors
            torch.save(valid_radii_pred, os.path.join(self.save_dir, "valid_radii_pred.pt"))
            torch.save(valid_radii_truth, os.path.join(self.save_dir, "valid_radii_truth.pt"))
            torch.save(valid_field_resim, os.path.join(self.save_dir, "valid_field_resim.pt"))
            torch.save(valid_field_truth, os.path.join(self.save_dir, "valid_field_truth.pt"))

        elif dataloader_idx == 1:  # train dataloader
            self.test_results['train']['radii_pred'].append(preds_real)
            self.test_results['train']['radii_truth'].append(radii)
            self.test_results['train']['field_resim'].append(resim_combined)
            self.test_results['train']['field_truth'].append(field_combined)

            # Convert to tensors and save as lists
            train_radii_pred = [pred for pred in self.test_results['train']['radii_pred']]
            train_radii_truth = [truth for truth in self.test_results['train']['radii_truth']]
            train_field_resim = [resim for resim in self.test_results['train']['field_resim']]
            train_field_truth = [truth for truth in self.test_results['train']['field_truth']]

            # Save as lists of tensors
            torch.save(train_radii_pred, os.path.join(self.save_dir, "train_radii_pred.pt"))
            torch.save(train_radii_truth, os.path.join(self.save_dir, "train_radii_truth.pt"))
            torch.save(train_field_resim, os.path.join(self.save_dir, "train_field_resim.pt"))
            torch.save(train_field_truth, os.path.join(self.save_dir, "train_field_truth.pt"))

            # train_radii_pred = np.concatenate([pred.cpu().numpy() for pred in self.test_results['train']['radii_pred']])
            # train_radii_truth = np.concatenate([truth.cpu().numpy() for truth in self.test_results['train']['radii_truth']])
            # np.savetxt(os.path.join(self.save_dir,"train_radii_pred.txt"), train_radii_pred)
            # np.savetxt(os.path.join(self.save_dir,"train_radii_truth.txt"), train_radii_truth)

            # train_field_resim = np.array(self.test_results['train']['field_resim'])
            # train_field_truth = np.array(self.test_results['train']['field_truth'])
            # np.save(os.path.join(self.save_dir, "train_field_resim.npy"), train_field_resim)
            # np.save(os.path.join(self.save_dir, "train_field_truth.npy"), train_field_truth)

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