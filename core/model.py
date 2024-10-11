#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import sys
import yaml
import torch
import numpy as np
from geomloss import SamplesLoss
from copy import deepcopy as copy
from torchmetrics import PeakSignalNoiseRatio
from torchvision.models import resnet50, resnet18, resnet34
import torch.nn as nn

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
sys.path.append("../")

from utils import parameter_manager
from pytorch_lightning import LightningModule


class FieldResponseModel(LightningModule):
    def __init__(self, params_model):
        super().__init__()
        
        self.params = params_model
        self.num_design_params = int(self.params['num_design_params'])
        self.learning_rate = self.params['learning_rate']
        
        # Build MLPs based on parameters
        self.mlp_real = self.build_mlp(self.num_design_params, self.params['mlp_real'])
        self.mlp_imag = self.build_mlp(self.num_design_params, self.params['mlp_imag'])
        
        self.save_hyperparameters()
        
        # Store necessary lists for tracking metrics
        self.train_nf_truth = []
        self.train_nf_pred = []
        self.val_nf_truth = []
        self.val_nf_pred = []
         
    def build_mlp(self, input_size, mlp_params):
        layers = []
        in_features = input_size
        for layer_size in mlp_params['layers']:
            layers.append(nn.Linear(in_features, layer_size))
            layers.append(self.get_activation_function(mlp_params['activation']))
            in_features = layer_size
        layers.append(nn.Linear(in_features, 166 * 166))  # 166x166 near field
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
        real_output = self.mlp_real(radii)
        imag_output = self.mlp_imag(radii)
        
        # Reshape to image size
        real_output = real_output.view(-1, 166, 166)
        imag_output = imag_output.view(-1, 166, 166)
        
        return real_output, imag_output
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def ae_loss(self, preds, labels, choice):
        if choice == 0:
            # MSE Loss
            preds = preds.to(torch.float32).contiguous()
            labels = labels.to(torch.float32).contiguous()
            fn = torch.nn.MSELoss()
            loss = fn(preds, labels)
        elif choice == 1:
            # Earth Mover's Distance / Sinkhorn
            preds = preds.to(torch.float64).contiguous()
            labels = labels.to(torch.float64).contiguous()
            fn = SamplesLoss("sinkhorn", p=1, blur=0.05)
            loss = fn(preds, labels)
            loss = torch.mean(loss)  # Aggregating the loss
        return loss
    
    def objective(self, batch, predictions):
        
        near_fields, radii = batch
        
        pred_real, pred_imag = predictions
        
        real_near_fields = near_fields[:, 0, :, :]
        imag_near_fields = near_fields[:, 1, :, :]

        # Near-field loss: compute separately for real and imaginary components
        near_field_loss_real = self.ae_loss(pred_real.squeeze(), real_near_fields, choice=0)
        near_field_loss_imag = self.ae_loss(pred_imag.squeeze(), imag_near_fields, choice=0)
        near_field_loss = near_field_loss_real + near_field_loss_imag
        
        return {"loss": near_field_loss}
    
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
        
        return {'loss': loss, 'output': preds, 'target': batch}
    
    def validation_step(self, batch, batch_idx):
        preds = self.shared_step(batch, batch_idx)
        loss_dict = self.objective(batch, preds)
        loss = loss_dict['loss']
        
        # log metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'loss': loss, 'output': preds, 'target': batch}
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        preds = self.shared_step(batch, batch_idx)
        
        self.organize_testing(preds, batch, batch_idx, dataloader_idx)
        
    def organize_testing(self, predictions, batch, batch_idx, dataloader_idx):
        pred_real, pred_imag = predictions
        near_fields, radii = batch
        real_near_fields, imag_near_fields = near_fields[:, 0, :, :], near_fields[:, 1, :, :]
        # Store predictions and ground truths for analysis after testing
        if dataloader_idx == 0: # val dataloader
            self.val_nf_truth.append(torch.stack([real_near_fields, imag_near_fields], dim=1).cpu().numpy())
            self.val_nf_pred.append(torch.stack([pred_real, pred_imag], dim=1).cpu().numpy())
        elif dataloader_idx == 1: # train dataloader
            self.train_nf_truth.append(torch.stack([real_near_fields, imag_near_fields], dim=1).cpu().numpy())
            self.train_nf_pred.append(torch.stack([pred_real, pred_imag], dim=1).cpu().numpy())
        else:
            raise ValueError(f"Invalid dataloader index: {dataloader_idx}")
        
    def on_test_end(self):
        # (near-field predictions and truths)
        val_results = {
            'nf_pred': np.concatenate([pred for pred in self.val_nf_pred]),
            'nf_truth': np.concatenate([truth for truth in self.val_nf_truth])
        }
        
        train_results = {
            'nf_pred': np.concatenate([pred for pred in self.train_nf_pred]),
            'nf_truth': np.concatenate([truth for truth in self.train_nf_truth])
        }

        self.logger.experiment.log_results(results=val_results, epoch=None, count=5, mode="val", name="results")
        self.logger.experiment.log_results(results=train_results, epoch=None, count=5, mode="train", name="results")