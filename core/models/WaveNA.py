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
from tqdm import tqdm
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
        self.automatic_optimization = False
        self.save_hyperparameters()
        
        # Store necessary lists for tracking metrics per fold
        self.test_results = {'train': {'radii_pred': [], 'radii_truth': [], 'field_resim': [], 'field_truth': []},
                            'valid': {'radii_pred': [], 'radii_truth': [],  'field_resim': [], 'field_truth': []}}
    
    
    def load_forward(self, checkpoint_path, config_path):
        forward_conf = load_config(config_path)
        model = WaveMLP(model_config=forward_conf.model)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device)['state_dict'])
        model.to(self.device)
        model.eval()
        return model
            
    def forward(self, input):
        # Inverse model: near_fields -> radii
        pred_near_fields = self.forward_model(input)
        # x now holds the design parameters optimized to produce near fields similar to target.
        return pred_near_fields
    
    def optimize_design(self, near_fields):
        batch_size = near_fields.shape[0]
        x = torch.rand((batch_size, self.num_design_conf), device=self.device, requires_grad=True)
        inner_optimizer = torch.optim.Adam([x], lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        for i in tqdm(range(self.na_iters)):
            inner_optimizer.zero_grad()
            # Forward pass through the frozen model (this now computes gradients w.r.t. x)
            pred_near_fields = self.forward_model(x)
            pred_near_fields.requires_grad_(True)
            fwd_pred_real = pred_near_fields.real
            fwd_pred_imag = pred_near_fields.imag
            fwd_pred_real.requires_grad_(True)
            fwd_pred_imag.requires_grad_(True)
            near_fields_real = near_fields[:, 0, :, :]
            near_fields_imag = near_fields[:, 1, :, :]
            near_fields_real.requires_grad_(True)
            near_fields_imag.requires_grad_(True)
            loss_real = loss_fn(fwd_pred_real, near_fields_real)
            loss_imag = loss_fn(fwd_pred_imag, near_fields_imag)
            loss = loss_real + loss_imag
            loss.requires_grad_(True)
            loss.backward()
            inner_optimizer.step()
            if i % 100 == 0:  # Log for first sample in batch
                self.log(f"na_inner_loss", loss, on_step=True)
                print(f"Iter {i}, Loss: {loss}")
        return x


    def training_step(self, batch, batch_idx):
        near_fields, radii = batch
        optimized_design = self.optimize_design(near_fields)
        optimized_design = optimized_design.to(torch.float32).contiguous()
        radii = radii.to(torch.float32).contiguous()
        fn = torch.nn.MSELoss()
        loss = fn(optimized_design, radii)
        # log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)        
        return None

    def validation_step(self, batch, batch_idx):
        near_fields, radii = batch
        optimized_design = self.optimize_design(near_fields)
        optimized_design = optimized_design.to(torch.float32).contiguous()
        radii = radii.to(torch.float32).contiguous()
        fn = torch.nn.MSELoss()
        loss = fn(optimized_design, radii)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return None
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        near_fields, radii = batch
        optimized_design = self.optimize_design(near_fields)
        self.organize_testing(optimized_design, batch, batch_idx, dataloader_idx)
        
    def configure_optimizers(self):
        return None
    
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