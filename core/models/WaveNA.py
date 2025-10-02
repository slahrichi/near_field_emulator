#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
from .CVNN import ComplexReLU, ModReLU
from .WaveMLP import WaveMLP
from conf.schema import load_config
import evaluation.evaluation as eval

sys.path.append("../")
torch.set_float32_matmul_precision('high')


class WaveNA(LightningModule):
    """Inverse design via the Neural Adjoint method (Ren et al. 2020)."""

    def __init__(self, model_config, fold_idx=None):
        super().__init__()

        self.conf = model_config
        self.learning_rate = self.conf.learning_rate
        self.na_iters = self.conf.na_iters
        self.K = self.conf.K
        self.lr_scheduler = getattr(self.conf, "lr_scheduler", "None")
        self.loss_func = self.conf.objective_function
        self.fold_idx = fold_idx
        self.name = self.conf.arch
        self.num_design_conf = int(self.conf.num_design_conf)
        self.forward_ckpt_path = self.conf.forward_ckpt_path
        self.forward_config_path = self.conf.forward_config_path
        self.radii_bounds = tuple(self.conf.radii_bounds)
        self.forward_model = self.load_forward(self.forward_ckpt_path, self.forward_config_path)
        for param in self.forward_model.parameters():
            param.requires_grad_(False)
        self.forward_model.eval()

        self.automatic_optimization = False

        lower = torch.full((self.num_design_conf,), self.radii_bounds[0], dtype=torch.float32)
        upper = torch.full((self.num_design_conf,), self.radii_bounds[1], dtype=torch.float32)
        self.register_buffer("radii_lower_bound", lower)
        self.register_buffer("radii_upper_bound", upper)
        self.register_buffer("radii_range", self.radii_upper_bound - self.radii_lower_bound)
        self.register_buffer("radii_mean", (self.radii_lower_bound + self.radii_upper_bound) / 2)

        self.inner_boundary_weight = 0.1

        self.save_hyperparameters()

        self.test_results = {
            'train': {'radii_pred': [], 'radii_truth': [], 'field_resim': [], 'field_truth': []},
            'valid': {'radii_pred': [], 'radii_truth': [], 'field_resim': [], 'field_truth': []}
        }

    def load_forward(self, checkpoint_path, config_path):
        if not checkpoint_path or not config_path:
            raise ValueError("Neural Adjoint requires forward_ckpt_path and forward_config_path in the config.")
        forward_conf = load_config(config_path)
        model = WaveMLP(model_config=forward_conf.model)
        state = torch.load(checkpoint_path, weights_only=False)
        state_dict = state['state_dict'] if isinstance(state, dict) and 'state_dict' in state else state
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def forward(self, near_fields):
        return self.optimize_design(near_fields)[0]

    def _ensure_forward_device(self):
        self.forward_model.to(self.device)

    def _run_forward(self, radii_batch):
        outputs = self.forward_model(radii_batch)
        if isinstance(outputs, tuple):
            pred_real, pred_imag = outputs
        elif torch.is_complex(outputs):
            pred_real = outputs.real
            pred_imag = outputs.imag
        else:
            raise ValueError("Unexpected forward model output type")
        return pred_real, pred_imag

    def _compute_candidate_losses(self, pred_real, pred_imag, target_real, target_imag, candidates, batch_size):
        mse_real = F.mse_loss(pred_real, target_real, reduction='none')
        mse_imag = F.mse_loss(pred_imag, target_imag, reduction='none')
        mse = (mse_real + mse_imag).flatten(1).mean(dim=1)

        deviation = torch.abs(candidates - self.radii_mean.unsqueeze(0)) - 0.5 * self.radii_range.unsqueeze(0)
        boundary_penalty = F.relu(deviation).sum(dim=1) * self.inner_boundary_weight

        total_candidate_loss = mse + boundary_penalty
        loss_per_target = total_candidate_loss.view(self.K, batch_size).mean(dim=0)
        total_loss = loss_per_target.mean()

        return total_loss, total_candidate_loss, mse.detach(), boundary_penalty.detach()

    def _initial_seed(self, batch_size, device, dtype):
        rand = torch.rand((self.K * batch_size, self.num_design_conf), device=device, dtype=dtype)
        seeds = self.radii_lower_bound.unsqueeze(0) + rand * self.radii_range.unsqueeze(0)
        return torch.nn.Parameter(seeds)

    def _step_scheduler(self, scheduler, loss):
        if scheduler is None:
            return
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss.detach())
        else:
            scheduler.step()

    def optimize_design(self, near_fields):
        if self.na_iters <= 0:
            raise ValueError("na_iters must be a positive integer")

        self._ensure_forward_device()
        device = near_fields.device
        dtype = near_fields.dtype
        batch_size = near_fields.shape[0]

        target_real = near_fields[:, 0].contiguous()
        target_imag = near_fields[:, 1].contiguous()

        target_real_rep = target_real.repeat_interleave(self.K, dim=0)
        target_imag_rep = target_imag.repeat_interleave(self.K, dim=0)

        candidates = self._initial_seed(batch_size, device, dtype)
        inner_optimizer = torch.optim.Adam([candidates], lr=self.learning_rate)

        scheduler = None
        if self.lr_scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(inner_optimizer, factor=0.5, patience=10, min_lr=1e-6)
        elif self.lr_scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(inner_optimizer, T_max=max(1, self.na_iters))

        last_mse = None
        last_candidate_loss = None

        grad_mode = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        try:
            with torch.inference_mode(False):
                with torch.enable_grad():
                    for iter_idx in range(self.na_iters):
                        inner_optimizer.zero_grad()
                        pred_real, pred_imag = self._run_forward(candidates)
                        total_loss, candidate_loss, mse, _ = self._compute_candidate_losses(
                            pred_real, pred_imag, target_real_rep, target_imag_rep, candidates, batch_size
                        )
                        total_loss.backward()
                    inner_optimizer.step()

                    with torch.no_grad():
                        candidates.data.clamp_(
                            self.radii_lower_bound.unsqueeze(0),
                            self.radii_upper_bound.unsqueeze(0)
                        )

                    self._step_scheduler(scheduler, total_loss)

                    last_mse = mse
                    last_candidate_loss = candidate_loss
        finally:
            torch.set_grad_enabled(grad_mode)

        candidates_final = candidates.detach().view(self.K, batch_size, self.num_design_conf)
        mse_matrix = last_mse.view(self.K, batch_size)
        best_indices = torch.argmin(mse_matrix, dim=0)

        gather_helper = torch.arange(batch_size, device=device)
        best_designs = candidates_final.permute(1, 0, 2)[gather_helper, best_indices]

        return best_designs, mse_matrix.detach(), last_candidate_loss.view(self.K, batch_size).detach()
        
    def training_step(self, batch, batch_idx):
        near_fields, radii = batch
        optimized_design, mse_matrix, _ = self.optimize_design(near_fields)

        optimized_design = optimized_design.to(torch.float32)
        radii = radii.to(torch.float32)

        loss = F.mse_loss(optimized_design, radii)
        per_sample_field_mse = mse_matrix.min(dim=0)[0].mean()

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_field_mse", per_sample_field_mse, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        near_fields, radii = batch
        optimized_design, mse_matrix, _ = self.optimize_design(near_fields)

        optimized_design = optimized_design.to(torch.float32)
        radii = radii.to(torch.float32)

        loss = F.mse_loss(optimized_design, radii)
        per_sample_field_mse = mse_matrix.min(dim=0)[0].mean()

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_field_mse", per_sample_field_mse, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        near_fields, radii = batch
        optimized_design, _, _ = self.optimize_design(near_fields)
        self.organize_testing(optimized_design, batch, batch_idx, dataloader_idx)
        
    def configure_optimizers(self):
        return []
    
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
        predictions = predictions.detach()

        self._ensure_forward_device()

        with torch.no_grad():
            pred_real, pred_imag = self._run_forward(predictions)

        if isinstance(pred_real, tuple):
            raise ValueError("Unexpected nested tuple from forward model")

        resim_combined = torch.stack([pred_real, pred_imag], dim=1).cpu().numpy()
        field_combined = near_fields.detach().cpu().numpy()

        if dataloader_idx == 0:
            store_key = 'valid'
        elif dataloader_idx == 1:
            store_key = 'train'
        else:
            raise ValueError(f"Invalid dataloader index: {dataloader_idx}")

        self.test_results[store_key]['radii_pred'].append(predictions.detach().cpu())
        self.test_results[store_key]['radii_truth'].append(radii.detach().cpu())
        self.test_results[store_key]['field_resim'].append(resim_combined)
        self.test_results[store_key]['field_truth'].append(field_combined)

    def on_test_end(self):
        # Concatenate results from all batches
        for mode in ['train', 'valid']:
            pred_entries = self.test_results[mode]['radii_pred']
            truth_entries = self.test_results[mode]['radii_truth']
            field_pred_entries = self.test_results[mode]['field_resim']
            field_truth_entries = self.test_results[mode]['field_truth']
            if isinstance(pred_entries, list) and pred_entries:
                self.test_results[mode]['radii_pred'] = np.concatenate([tensor.cpu().detach().numpy() for tensor in pred_entries], axis=0)
            if isinstance(truth_entries, list) and truth_entries:
                self.test_results[mode]['radii_truth'] = np.concatenate([tensor.cpu().detach().numpy() for tensor in truth_entries], axis=0)
            if isinstance(field_pred_entries, list) and field_pred_entries:
                self.test_results[mode]['field_resim'] = np.concatenate(field_pred_entries, axis=0)
            if isinstance(field_truth_entries, list) and field_truth_entries:
                self.test_results[mode]['field_truth'] = np.concatenate(field_truth_entries, axis=0)

        for dataset in ['train', 'valid']:
            entries = self.test_results[dataset]['radii_pred']
            has_data = isinstance(entries, np.ndarray) and entries.size > 0
            if not has_data:
                continue
            eval.metrics(self.test_results, dataset=dataset, save_fig=True, save_dir='results/meep_meep/', plot_mse=False)
