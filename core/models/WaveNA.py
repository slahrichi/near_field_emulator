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
import evaluation.evaluation as eval

sys.path.append("../")
torch.set_float32_matmul_precision('high')  

class WaveNA(LightningModule):
    """Radii Prediction Model  
    Architecture: MLP  
    Modes: Neural Adjoint (Ren et al. 2020)"""
    def __init__(self, model_config, fold_idx=None):
        super().__init__()
        
        self.conf = model_config
        self.learning_rate = self.conf.learning_rate
        self.na_iters = self.conf.na_iters
        self.K = self.conf.K
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
        self.radii_lower_bound = torch.tensor([self.radii_bounds[0]] * self.num_design_conf, device=self.device, dtype=torch.float32)
        self.radii_upper_bound = torch.tensor([self.radii_bounds[1]] * self.num_design_conf, device=self.device, dtype=torch.float32)
        self.register_buffer('radii_range', self.radii_upper_bound - self.radii_lower_bound)
        self.register_buffer('radii_mean', (self.radii_lower_bound + self.radii_upper_bound) / 2)
            
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
    
    def make_loss(self, pred_near_fields, near_fields_repeated, x):
        fwd_pred_real = pred_near_fields.real
        fwd_pred_imag = pred_near_fields.imag
        near_fields_real = near_fields_repeated[:, 0, :, :]
        near_fields_imag = near_fields_repeated[:, 1, :, :]

        loss_fn = nn.MSELoss()
        loss_real = loss_fn(fwd_pred_real, near_fields_real)
        loss_imag = loss_fn(fwd_pred_imag, near_fields_imag)
        mse_loss = loss_real + loss_imag

        relu = torch.nn.ReLU()
        bdy_loss_all = relu(torch.abs(x - self.radii_mean) - 0.5 * self.radii_range)
        bdy_loss = torch.sum(bdy_loss_all) * 0.10
        total_loss = mse_loss + bdy_loss
        total_loss.requires_grad_(True)
        return total_loss    
            
    def optimize_design(self, near_fields):
        batch_size = near_fields.shape[0]
        K = self.K
        near_fields_repeated = near_fields.repeat(K, 1, 1, 1)
        x = torch.rand((K*batch_size, self.num_design_conf), device=self.device, requires_grad=True)
        inner_optimizer = torch.optim.Adam([x], lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(inner_optimizer, T_max=self.na_iters)

        for i in tqdm(range(self.na_iters)):
            inner_optimizer.zero_grad()
            # Forward pass through the frozen model (this now computes gradients w.r.t. x)
            pred_near_fields = self.forward_model(x)
            pred_near_fields.requires_grad_(True)
            loss = self.make_loss(pred_near_fields, near_fields_repeated, x)            
            loss.backward()
            inner_optimizer.step()
            scheduler.step()
            # TODO: N_ITERS 2x what's needed to go down
            if i % 100 == 0:  # Log for first sample in batch
                self.log(f"na_inner_loss", loss, on_step=True)
                print(f"Iter {i}, Loss: {loss}")
        
        x_optimized = x.view(K, batch_size, self.num_design_conf)
        x_optimized_flat = x_optimized.view(K * batch_size, self.num_design_conf)
        pred_near_fields = self.forward_model(x_optimized_flat)
        pred_real = pred_near_fields.real.view(K, batch_size, *pred_near_fields.shape[1:])
        pred_imag = pred_near_fields.imag.view(K, batch_size, *pred_near_fields.shape[1:])
        near_fields_real = near_fields[:, 0, :, :].unsqueeze(0).repeat(K, 1, 1, 1)
        near_fields_imag = near_fields[:, 1, :, :].unsqueeze(0).repeat(K, 1, 1, 1)
        mse_real = torch.mean((pred_real - near_fields_real) ** 2, dim=[2, 3])
        mse_imag = torch.mean((pred_imag - near_fields_imag) ** 2, dim=[2, 3])
        mse_total = mse_real + mse_imag  # Shape: (K, batch_size)
        min_indices = torch.argmin(mse_total, dim=0)  # Shape: (batch_size,)
        batch_indices = torch.arange(batch_size)
        best_x = x_optimized[min_indices, batch_indices, :]  # Shape: (batch_size, num_design_conf)
        
        return best_x
        
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
        eval.metrics(self.test_results, dataset='train', save_fig=False, save_dir='results/meep_meep/', plot_mse=False)
        eval.metrics(self.test_results, dataset='valid', save_fig=False, save_dir='results/meep_meep/', plot_mse=False)
