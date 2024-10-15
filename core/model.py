#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import sys
import yaml
import torch
import numpy as np
from geomloss import SamplesLoss
from copy import deepcopy as copy
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.models import resnet50, resnet18, resnet34
import torch.nn as nn
from pytorch_lightning import LightningModule
import os
import pickle
import math

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
sys.path.append("../")

from utils import parameter_manager


class FieldResponseModel(LightningModule):
    def __init__(self, params_model, fold_idx=None):
        super().__init__()
        
        self.params = params_model
        self.num_design_params = int(self.params['num_design_params'])
        self.learning_rate = self.params['learning_rate']
        self.loss_func = self.params['objective_function']
        self.fold_idx = fold_idx
        
        # patch test
        self.patch_mlps = self.params['patch_mlps'] # whether or not to split things up
        if self.patch_mlps:
            self.patch_size = 16
            self.num_patches_height = math.ceil(166 / self.patch_size)
            self.num_patches_width = math.ceil(166 / self.patch_size)
            self.num_patches = self.num_patches_height * self.num_patches_width
            # Build MLPs for each patch
            self.mlp_real = nn.ModuleList([
                self.build_mlp(self.num_design_params, self.params['mlp_real']) for _ in range(self.num_patches)
            ])
            self.mlp_imag = nn.ModuleList([
                self.build_mlp(self.num_design_params, self.params['mlp_imag']) for _ in range(self.num_patches)
        ])
        else:
            # Build full MLPs
            self.mlp_real = self.build_mlp(self.num_design_params, self.params['mlp_real'])
            self.mlp_imag = self.build_mlp(self.num_design_params, self.params['mlp_imag'])
        
        self.save_hyperparameters()
        
        # Store necessary lists for tracking metrics per fold
        self.test_results = {'train': {'nf_pred': [], 'nf_truth': []},
                             'valid': {'nf_pred': [], 'nf_truth': []}}
        
        # Initialize metrics
        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
         
    def build_mlp(self, input_size, mlp_params, patches=False):
        layers = []
        in_features = input_size
        for layer_size in mlp_params['layers']:
            layers.append(nn.Linear(in_features, layer_size))
            layers.append(self.get_activation_function(mlp_params['activation']))
            in_features = layer_size
        if self.patch_mlps: # just a single patch
            output_size = self.patch_size * self.patch_size  
        else: # 166x166 near field
            output_size = 166 * 166
        layers.append(nn.Linear(in_features, output_size))  
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
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")
        
    def forward(self, radii):
        # multiple sets of MLPs for our patches
        if self.patch_mlps:
            batch_size = radii.size(0)
            real_patches = []
            imag_patches = []

            for i in range(self.num_patches):
                # Real part
                real_patch = self.mlp_real[i](radii)
                real_patch = real_patch.view(batch_size, self.patch_size, self.patch_size)
                real_patches.append(real_patch)

                # Imaginary part
                imag_patch = self.mlp_imag[i](radii)
                imag_patch = imag_patch.view(batch_size, self.patch_size, self.patch_size)
                imag_patches.append(imag_patch)

            # Assemble patches
            real_output = self.assemble_patches(real_patches, batch_size)
            imag_output = self.assemble_patches(imag_patches, batch_size)

            # Crop to original size if necessary
            real_output = real_output[:, :166, :166]
            imag_output = imag_output[:, :166, :166]
        else:
            # grab output directly
            real_output = self.mlp_real(radii)
            imag_output = self.mlp_imag(radii)
            # Reshape to image size
            real_output = real_output.view(-1, 166, 166)
            imag_output = imag_output.view(-1, 166, 166)
        
        return real_output, imag_output
        
    def assemble_patches(self, patches, batch_size):
        # reshape patches into grid
        patches_per_row = self.num_patches_width
        patches_per_col = self.num_patches_height
        patch_size = self.patch_size

        patches_tensor = torch.stack(patches, dim=1)  # Shape: [batch_size, num_patches, patch_size, patch_size]
        patches_tensor = patches_tensor.view(
            batch_size,
            patches_per_col,
            patches_per_row,
            patch_size,
            patch_size
        )

        # permute and reshape to assemble the image
        output = patches_tensor.permute(0, 1, 3, 2, 4).contiguous()
        output = output.view(
            batch_size,
            patches_per_col * patch_size,
            patches_per_row * patch_size
        )

        return output 
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                    factor=0.1, patience=5, 
                                                                    verbose=True, min_lr=1e-6,
                                                                    threshold=0.001),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1,
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def compute_loss(self, preds, labels, choice):
        if choice == 'mse':
            # Mean Squared Error
            preds = preds.to(torch.float32).contiguous()
            labels = labels.to(torch.float32).contiguous()
            fn = torch.nn.MSELoss()
            loss = fn(preds, labels)
        elif choice == 'emd':
            # Earth Mover's Distance / Sinkhorn
            preds = preds.to(torch.float64).contiguous()
            labels = labels.to(torch.float64).contiguous()
            fn = SamplesLoss("sinkhorn", p=1, blur=0.05)
            loss = fn(preds, labels)
            loss = torch.mean(loss)  # Aggregating the loss
        elif choice == 'psnr':
            # Peak Signal-to-Noise Ratio
            preds, labels = preds.unsqueeze(1), labels.unsqueeze(1)  # channel dim
            fn = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
            loss = fn(preds, labels)
        elif choice == 'ssim':
            # Structural Similarity Index
            preds, labels = preds.unsqueeze(1), labels.unsqueeze(1)  # channel dim
            torch.use_deterministic_algorithms(True, warn_only=True)
            with torch.backends.cudnn.flags(enabled=False):
                fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
                ssim_value = fn(preds, labels)
                loss = 1 - ssim_value  # SSIM is a similarity metric
        else:
            raise ValueError(f"Unsupported loss function: {choice}")
            
        return loss
    
    def objective(self, batch, predictions):
        near_fields, radii = batch
        
        pred_real, pred_imag = predictions
        
        real_near_fields = near_fields[:, 0, :, :]
        imag_near_fields = near_fields[:, 1, :, :]

        # Near-field loss: compute separately for real and imaginary components
        near_field_loss_real = self.compute_loss(pred_real.squeeze(), real_near_fields, choice=self.loss_func)
        near_field_loss_imag = self.compute_loss(pred_imag.squeeze(), imag_near_fields, choice=self.loss_func)
        near_field_loss = near_field_loss_real + near_field_loss_imag
        
        # compute other metrics for logging besides specified loss function
        choices = {
            'mse': None,
            'emd': None,
            'ssim': None,
            'psnr': None
        }
        
        for key in choices:
            if key != self.loss_func:
                loss_real = self.compute_loss(pred_real.squeeze(), real_near_fields, choice=key)
                loss_imag = self.compute_loss(pred_imag.squeeze(), imag_near_fields, choice=key)
                loss = loss_real + loss_imag
                choices[key] = loss
        
        return {"loss": near_field_loss, **choices}
    
    def shared_step(self, batch, batch_idx):
        near_fields, radii = batch
        
        pred_real, pred_imag = self.forward(radii)
        
        return pred_real, pred_imag
    
    def training_step(self, batch, batch_idx):
        preds = self.shared_step(batch, batch_idx)
        loss_dict = self.objective(batch, preds)
        loss = loss_dict['loss']
        
        # log metrics
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
        
        #print(f"keys: {loss_dict.keys()}")
        
        # log metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        other_metrics = [f"{key}" for key in loss_dict.keys() if key != 'loss' and key != self.loss_func]
        for key in other_metrics:
            #print(f"valid_{key}", loss_dict[key])
            self.log(f"valid_{key}", loss_dict[key], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'loss': loss, 'output': preds, 'target': batch}
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        preds = self.shared_step(batch, batch_idx)   
        self.organize_testing(preds, batch, batch_idx, dataloader_idx)
        
    def organize_testing(self, predictions, batch, batch_idx, dataloader_idx):
        pred_real, pred_imag = predictions
        near_fields, radii = batch
        real_near_fields, imag_near_fields = near_fields[:, 0, :, :], near_fields[:, 1, :, :]
        
        # collect preds and ground truths
        preds_combined = torch.stack([pred_real, pred_imag], dim=1).cpu().numpy()
        truths_combined = torch.stack([real_near_fields, imag_near_fields], dim=1).cpu().numpy()
        
        # Store predictions and ground truths for analysis after testing
        if dataloader_idx == 0:  # val dataloader
            self.test_results['valid']['nf_pred'].append(preds_combined)
            self.test_results['valid']['nf_truth'].append(truths_combined)
        elif dataloader_idx == 1:  # train dataloader
            self.test_results['train']['nf_pred'].append(preds_combined)
            self.test_results['train']['nf_truth'].append(truths_combined)
        else:
            raise ValueError(f"Invalid dataloader index: {dataloader_idx}")
        
    def on_test_end(self):
        # Concatenate results from all batches
        for mode in ['train', 'valid']:
            self.test_results[mode]['nf_pred'] = np.concatenate(self.test_results[mode]['nf_pred'], axis=0)
            self.test_results[mode]['nf_truth'] = np.concatenate(self.test_results[mode]['nf_truth'], axis=0)
       
            # Ensure fold index is valid
            if self.fold_idx is not None:
                fold_suffix = f"_fold{self.fold_idx+1}"
            else:
                raise ValueError("fold_idx is not set!")

            # Log results for the current fold
            name = f"results{fold_suffix}"
            self.logger.experiment.log_results(
                results=self.test_results[mode],
                epoch=None,
                mode=mode,
                name=name
            )