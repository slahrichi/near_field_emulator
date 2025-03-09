#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
from .CVNN import ComplexReLU, ModReLU, ComplexLinearFinal
from .WaveMLP import WaveMLP
from conf.schema import load_config

sys.path.append("../")

class WaveNA(LightningModule):
    """Radii Prediction Model  
    Architecture: MLP  
    Modes: Neural Adjoint (Ren et al. 2020)"""
    def __init__(self, model_config, fold_idx=None):
        super().__init__()
        
        self.conf = model_config
        self.learning_rate = self.conf.learning_rate
        self.na_iters = self.conf.na_iters
#        self.lr_scheduler = self.conf.lr_scheduler
        self.loss_func = self.conf.objective_function
        self.fold_idx = fold_idx
        self.name = self.conf.arch # inverse
        self.num_design_conf = int(self.conf.num_design_conf)
        self.forward_ckpt_path = self.conf.forward_ckpt_path
        self.forward_config_path = self.conf.forward_config_path
        self.radii_bounds = self.conf.radii_bounds
        self.forward_model = self.load_forward(self.forward_ckpt_path, self.forward_config_path)
        for param in self.forward_model.parameters():
            param.requires_grad = False
        self.forward_model.eval()


        self.save_hyperparameters()
        
        # Store necessary lists for tracking metrics per fold
        self.test_results = {'train': {'radii_pred': [], 'radii_truth': []},
                            'valid': {'radii_pred': [], 'radii_truth': []}}
    
    
    def load_forward(self, checkpoint_path, config_path):
        forward_conf = load_config(config_path)
        model = WaveMLP(model_config=forward_conf.model)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device)['state_dict'])
        model.to(self.device)
        model.eval()

        return model
            
    def forward(self, input):
        # Inverse model: near_fields -> radii
        near_fields = input
        batch_size = near_fields.shape[0]
        x = torch.rand((batch_size, self.num_design_conf), device=self.device, requires_grad=True)
        
        optimizer = optim.Adam([x], lr=self.inner_lr)
        loss_fn = nn.MSELoss()

        for i in range(self.na_iters):
            optimizer.zero_grad()
            # Get predicted near fields from the forward model given current design parameters x.
            pred_near_fields = self.forward_model(x)
            # Compute loss between the predicted and target near fields.
            loss = loss_fn(pred_near_fields, near_fields)
            loss.backward()
            optimizer.step()

            # Enforce bounds on x
            x.data.clamp_(self.design_bounds[0], self.design_bounds[1])

            if i % 100 == 0:
                self.log("na_inner_loss", loss, prog_bar=True)
                
        # x now holds the design parameters optimized to produce near fields similar to target.
        return x
    
    def training_step(self, batch, batch_idx):
        near_fields, radii = batch
        preds = self.forward(near_fields)
        preds = preds.to(torch.float32).contiguous()
        radii = radii.to(torch.float32).contiguous()
        fn = torch.nn.MSELoss()
        loss = fn(preds, radii)
        # log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)        
        return {'loss': loss, 'output': preds, 'target': batch}

    def validation_step(self, batch, batch_idx):
        near_fields, radii = batch
        preds = self.forward(near_fields)
        preds = preds.to(torch.float32).contiguous()
        radii = radii.to(torch.float32).contiguous()
        fn = torch.nn.MSELoss()
        loss = fn(preds, radii)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return {'loss': loss, 'output': preds, 'target': batch}
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        near_fields, radii = batch
        preds = self.forward(near_fields)
        self.organize_testing(preds, batch, batch_idx, dataloader_idx)
        
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

    def organize_testing(self, predictions, batch, batch_idx, dataloader_idx):
        near_fields, radii = batch
        if self.name == 'inverse':
            # Saving real part of the prediction only; double check that imaginary part can be disregarded
            preds_real = predictions.real
            # Store predictions and ground truths for analysis after testing
            if dataloader_idx == 0:  # val dataloader
                self.test_results['valid']['radii_pred'].append(preds_real)
                self.test_results['valid']['radii_truth'].append(radii)
            elif dataloader_idx == 1:  # train dataloader
                self.test_results['train']['radii_pred'].append(preds_real)
                self.test_results['train']['radii_truth'].append(radii)
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