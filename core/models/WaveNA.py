#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import copy
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
        optimizer_name = getattr(self.conf, "optimizer", "adam")
        self.inner_optimizer_name = optimizer_name.lower() if isinstance(optimizer_name, str) else "adam"
        seed_strategy = getattr(self.conf, "na_seed_strategy", "random")
        self.na_seed_strategy = seed_strategy.lower() if isinstance(seed_strategy, str) else "random"
        if self.na_seed_strategy not in {"random", "ground_truth", "jitter"}:
            raise ValueError(
                f"Unsupported na_seed_strategy '{seed_strategy}'. Choose from 'random', 'ground_truth', or 'jitter'."
            )
        self.na_seed_jitter_std = float(getattr(self.conf, "na_seed_jitter_std", 0.01))
        self.na_track_stabilization = bool(getattr(self.conf, "na_track_stabilization", False))
        self.na_stabilization_tol = float(getattr(self.conf, "na_stabilization_tol", 1e-4))
        self.na_enable_early_exit = bool(getattr(self.conf, "na_enable_early_exit", False))
        self.na_early_exit_tol = float(getattr(self.conf, "na_early_exit_tol", 1e-3))
        patience = int(getattr(self.conf, "na_early_exit_patience", 1))
        self.na_early_exit_patience = max(1, patience)
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
        self.latest_debug_info = {}
        self.debug_history = {'train': [], 'val': [], 'test': []}

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

    def _initial_seed(self, batch_size, device, dtype, target_radii=None):
        lower = self.radii_lower_bound.to(device=device, dtype=dtype).unsqueeze(0)
        upper = self.radii_upper_bound.to(device=device, dtype=dtype).unsqueeze(0)
        range_tensor = self.radii_range.to(device=device, dtype=dtype).unsqueeze(0)

        if self.na_seed_strategy == 'random':
            rand = torch.rand((self.K * batch_size, self.num_design_conf), device=device, dtype=dtype)
            initial_values = lower + rand * range_tensor
        else:
            if target_radii is None:
                raise ValueError(
                    f"Seed strategy '{self.na_seed_strategy}' requires ground-truth radii, but none were provided."
                )
            base = target_radii.to(device=device, dtype=dtype)
            initial_values = base.repeat_interleave(self.K, dim=0)
            if self.na_seed_strategy == 'jitter' and self.na_seed_jitter_std > 0:
                noise = torch.randn_like(initial_values) * self.na_seed_jitter_std
                initial_values = initial_values + noise
            initial_values = initial_values.clamp(lower, upper)

        initial_values = initial_values.clamp(lower, upper)

        seeds = torch.nn.Parameter(initial_values)
        seed_snapshot = initial_values.detach().clone()
        return seeds, seed_snapshot

    def _log_na_debug_metrics(self, debug_info, stage_prefix="train"):
        if not isinstance(debug_info, dict) or not debug_info:
            return

        def _safe_tensor(value):
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                return value.detach().to(device='cpu', dtype=torch.float32)
            return None

        stabilization_vals = _safe_tensor(debug_info.get('stabilization_iters'))
        if stabilization_vals is not None:
            self.log(
                f"{stage_prefix}_na_stabilization_iter_mean",
                stabilization_vals.mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True
            )

        best_disp = _safe_tensor(debug_info.get('best_displacement'))
        if best_disp is not None:
            self.log(
                f"{stage_prefix}_na_best_disp_mean",
                best_disp.mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True
            )
            self.log(
                f"{stage_prefix}_na_best_disp_max",
                best_disp.max(),
                on_step=False,
                on_epoch=True,
                sync_dist=True
            )

        initial_best_mse = _safe_tensor(debug_info.get('initial_best_mse'))
        if initial_best_mse is not None:
            self.log(
                f"{stage_prefix}_na_initial_best_mse",
                initial_best_mse.mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True
            )

        final_target_mse = _safe_tensor(debug_info.get('final_target_mse'))
        if final_target_mse is not None:
            self.log(
                f"{stage_prefix}_na_final_best_mse",
                final_target_mse.mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True
            )

        early_exit_iter = debug_info.get('early_exit_iter')
        if isinstance(early_exit_iter, int):
            self.log(
                f"{stage_prefix}_na_early_exit_iter",
                float(early_exit_iter),
                on_step=False,
                on_epoch=True,
                sync_dist=True
            )

    def _store_debug_info(self, debug_info, stage):
        if not isinstance(debug_info, dict) or stage not in self.debug_history:
            return
        snapshot = {}
        for key, value in debug_info.items():
            if isinstance(value, torch.Tensor):
                snapshot[key] = value.detach().cpu()
            else:
                snapshot[key] = value
        self.debug_history[stage].append(snapshot)

    def get_debug_history(self, stage=None):
        if stage is None:
            return {name: [self._clone_debug_entry(entry) for entry in entries]
                    for name, entries in self.debug_history.items()}
        if stage not in self.debug_history:
            raise ValueError(f"Unknown debug history stage '{stage}'.")
        return [self._clone_debug_entry(entry) for entry in self.debug_history[stage]]

    def _clone_debug_entry(self, entry):
        cloned = {}
        for key, value in entry.items():
            if isinstance(value, torch.Tensor):
                cloned[key] = value.clone()
            else:
                cloned[key] = copy.deepcopy(value)
        return cloned

    def _build_inner_optimizer(self, candidates):
        if self.inner_optimizer_name == 'adam':
            return torch.optim.Adam([candidates], lr=self.learning_rate)
        if self.inner_optimizer_name == 'sgd':
            return torch.optim.SGD([candidates], lr=self.learning_rate)

        requested = getattr(self.conf, "optimizer", self.inner_optimizer_name)
        raise ValueError(
            f"Unsupported inner optimizer '{requested}'. Supported options are 'adam' and 'sgd'."
        )

    def _step_scheduler(self, scheduler, loss):
        if scheduler is None:
            return
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss.detach())
        else:
            scheduler.step()

    def optimize_design(self, near_fields, target_radii=None):
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

        lower_bound_tensor = self.radii_lower_bound.to(device=device, dtype=dtype).unsqueeze(0)
        upper_bound_tensor = self.radii_upper_bound.to(device=device, dtype=dtype).unsqueeze(0)

        if target_radii is not None:
            target_radii = target_radii.to(device=device, dtype=dtype)

        candidates, initial_snapshot = self._initial_seed(batch_size, device, dtype, target_radii)
        initial_snapshot = initial_snapshot.view(self.K, batch_size, self.num_design_conf)
        initial_candidates = initial_snapshot
        inner_optimizer = self._build_inner_optimizer(candidates)

        scheduler = None
        if self.lr_scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(inner_optimizer, factor=0.5, patience=10, min_lr=1e-6)
        elif self.lr_scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(inner_optimizer, T_max=max(1, self.na_iters))

        last_mse = None
        last_candidate_loss = None
        last_total_loss = None
        last_displacement = None

        debug_info = {}

        stabilization_iters = None
        stabilized_mask = None
        prev_best_mse = None
        if self.na_track_stabilization:
            stabilization_iters = torch.full((batch_size,), self.na_iters, device=device, dtype=torch.int64)
            stabilized_mask = torch.zeros_like(stabilization_iters, dtype=torch.bool)
            prev_best_mse = None

        early_exit_counter = 0
        early_exit_iter = None

        if target_radii is not None:
            target_repeat = target_radii.unsqueeze(0).expand(self.K, batch_size, self.num_design_conf)
            initial_mse = F.mse_loss(initial_snapshot, target_repeat, reduction='none').flatten(2).mean(dim=2)
            debug_info['initial_best_mse'] = initial_mse.min(dim=0)[0].detach()

        gather_helper = torch.arange(batch_size, device=device)

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
                            candidates.data.clamp_(lower_bound_tensor, upper_bound_tensor)

                        mse_matrix_iter = mse.view(self.K, batch_size)
                        best_mse_iter, best_indices_iter = mse_matrix_iter.min(dim=0)

                        last_mse = mse.detach()
                        last_candidate_loss = candidate_loss.detach()
                        last_total_loss = total_loss.detach()

                        if self.na_track_stabilization:
                            if prev_best_mse is not None:
                                improvement = torch.abs(prev_best_mse - best_mse_iter.detach())
                                newly_stabilized = (improvement <= self.na_stabilization_tol) & (~stabilized_mask)
                                stabilization_iters[newly_stabilized] = iter_idx
                                stabilized_mask |= newly_stabilized
                            prev_best_mse = best_mse_iter.detach()

                        if self.na_enable_early_exit:
                            candidates_view = candidates.detach().view(self.K, batch_size, self.num_design_conf)
                            best_design_iter = candidates_view.permute(1, 0, 2)[gather_helper, best_indices_iter]
                            initial_best_iter = initial_candidates.permute(1, 0, 2)[gather_helper, best_indices_iter]
                            displacement_iter = (best_design_iter - initial_best_iter).abs().max(dim=1)[0]
                            last_displacement = displacement_iter.detach()
                            if displacement_iter.max() <= self.na_early_exit_tol:
                                early_exit_counter += 1
                                if early_exit_counter >= self.na_early_exit_patience:
                                    early_exit_iter = iter_idx
                                    break
                            else:
                                early_exit_counter = 0

            if scheduler is not None and last_total_loss is not None:
                self._step_scheduler(scheduler, last_total_loss)
        finally:
            torch.set_grad_enabled(grad_mode)

        candidates_final = candidates.detach().view(self.K, batch_size, self.num_design_conf)
        mse_matrix = last_mse.view(self.K, batch_size)
        best_indices = torch.argmin(mse_matrix, dim=0)

        best_designs = candidates_final.permute(1, 0, 2)[gather_helper, best_indices]

        if stabilization_iters is not None:
            debug_info['stabilization_iters'] = stabilization_iters.detach()

        initial_best_designs = initial_candidates.permute(1, 0, 2)[gather_helper, best_indices]
        debug_info['best_displacement'] = (best_designs - initial_best_designs).abs().max(dim=1)[0].detach()

        if target_radii is not None:
            debug_info['final_target_mse'] = F.mse_loss(
                best_designs, target_radii, reduction='none'
            ).flatten(1).mean(dim=1).detach()

        if last_displacement is not None:
            debug_info['last_iter_displacement'] = last_displacement
        if early_exit_iter is not None:
            debug_info['early_exit_iter'] = early_exit_iter

        self.latest_debug_info = debug_info

        return best_designs, mse_matrix.detach(), last_candidate_loss.view(self.K, batch_size).detach(), debug_info
        
    def training_step(self, batch, batch_idx):
        near_fields, radii = batch
        optimized_design, mse_matrix, _, debug_info = self.optimize_design(near_fields, target_radii=radii)

        optimized_design = optimized_design.to(torch.float32)
        radii = radii.to(torch.float32)

        loss = F.mse_loss(optimized_design, radii)
        per_sample_field_mse = mse_matrix.min(dim=0)[0].mean()

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_field_mse", per_sample_field_mse, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        self._log_na_debug_metrics(debug_info, stage_prefix="train")
        self._store_debug_info(debug_info, 'train')

        return loss

    def validation_step(self, batch, batch_idx):
        near_fields, radii = batch
        optimized_design, mse_matrix, _, debug_info = self.optimize_design(near_fields, target_radii=radii)

        optimized_design = optimized_design.to(torch.float32)
        radii = radii.to(torch.float32)

        loss = F.mse_loss(optimized_design, radii)
        per_sample_field_mse = mse_matrix.min(dim=0)[0].mean()

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_field_mse", per_sample_field_mse, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        self._log_na_debug_metrics(debug_info, stage_prefix="val")
        self._store_debug_info(debug_info, 'val')

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        near_fields, radii = batch
        optimized_design, _, _, debug_info = self.optimize_design(near_fields, target_radii=radii)
        self._log_na_debug_metrics(debug_info, stage_prefix="test")
        self._store_debug_info(debug_info, 'test')
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
