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
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning import LightningModule
import math

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
#from utils import parameter_manager
from core.ConvLSTM import ConvLSTM
from core.autoencoder import Encoder, Decoder
sys.path.append("../")


class WaveMLP(LightningModule):
    """Near Field Response Prediction Model  
    Architecture: MLPs (real and imaginary)  
    Modes: Full, patch-wise"""
    def __init__(self, params_model, fold_idx=None):
        super().__init__()
        
        self.params = params_model
        self.num_design_params = int(self.params['num_design_params'])
        self.learning_rate = self.params['learning_rate']
        self.lr_scheduler = self.params['lr_scheduler']
        self.loss_func = self.params['objective_function']
        self.fold_idx = fold_idx
        self.approach = self.params['mlp_strategy']
        self.patch_size = self.params['patch_size'] # for non-default approaches
        
        if self.approach == 1: 
            # patch approach
            self.output_size = (self.patch_size)**2
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
        elif self.approach == 2: 
            # distributed subset approach
            self.output_size = (self.patch_size)**2
            # build MLPs
            self.mlp_real = self.build_mlp(self.num_design_params, self.params['mlp_real'])
            self.mlp_imag = self.build_mlp(self.num_design_params, self.params['mlp_imag'])
            
        else:
            # Build full MLPs
            self.output_size = 166 * 166
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
        layers.append(nn.Linear(in_features, self.output_size))  
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
        if self.approach == 1:
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
        elif self.approach == 2:
            # grab output directly
            real_output = self.mlp_real(radii)
            imag_output = self.mlp_imag(radii)
            # reshape to patch_size x patch_size    
            real_output = real_output.view(-1, self.patch_size, self.patch_size)
            imag_output = imag_output.view(-1, self.patch_size, self.patch_size)
        else:
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
        # setup specified scheduler
        if self.lr_scheduler == 'ReduceLROnPlateau':
            choice = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                    factor=0.5, patience=5, 
                                                                    min_lr=1e-6, threshold=0.001, 
                                                                    cooldown=2)
        elif self.lr_scheduler == 'CosineAnnealingLR':
            choice = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                T_max=self.params['num_epochs'])
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
    
    def compute_loss(self, preds, labels, choice):
        if choice == 'mse':
            # Mean Squared Error
            preds = preds.to(torch.float32).contiguous()
            labels = labels.to(torch.float32).contiguous()
            fn = torch.nn.MSELoss()
            loss = fn(preds, labels)
        elif choice == 'emd':
            # ignoring emd for now
            raise NotImplementedError("Earth Mover's Distance not implemented!")
            '''# Earth Mover's Distance / Sinkhorn
            preds = preds.to(torch.float64).contiguous()
            labels = labels.to(torch.float64).contiguous()
            fn = SamplesLoss("sinkhorn", p=1, blur=0.05)
            loss = fn(preds, labels)
            loss = torch.mean(loss)  # Aggregating the loss'''
        elif choice == 'psnr':
            # Peak Signal-to-Noise Ratio
            preds, labels = preds.unsqueeze(1), labels.unsqueeze(1)  # channel dim
            fn = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
            loss = fn(preds, labels)
        elif choice == 'ssim':
            # Structural Similarity Index
            if preds.size(-1) < 11 or preds.size(-2) < 11:
                loss = 0 # if the size is too small, SSIM is not defined
            else:
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
            #'emd': None,
            'ssim': None,
            'psnr': None
        }
        
        for key in choices:
            if key != self.loss_func:
                # ignoring emd for now - geomloss library has issues
                '''if key == 'emd':
                    # Reshape tensors to (batch_size, num_pixels, 1)
                    pred_real_reshaped = pred_real.view(pred_real.size(0), -1, 1)
                    real_near_fields_reshaped = real_near_fields.view(real_near_fields.size(0), -1, 1)
                    pred_imag_reshaped = pred_imag.view(pred_imag.size(0), -1, 1)
                    imag_near_fields_reshaped = imag_near_fields.view(imag_near_fields.size(0), -1, 1)

                    loss_real = self.compute_loss(pred_real_reshaped, real_near_fields_reshaped, choice=key)
                    loss_imag = self.compute_loss(pred_imag_reshaped, imag_near_fields_reshaped, choice=key)
                else:'''
                loss_real = self.compute_loss(pred_real, real_near_fields, choice=key)
                loss_imag = self.compute_loss(pred_imag, imag_near_fields, choice=key)
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
            

class WaveLSTM(LightningModule):
    """Near Field Response Time Series Prediction Model  
    Architecture: LSTM"""
    def __init__(self, params_model, fold_idx=None):
        super().__init__()
        
        self.params = params_model
        self.num_design_params = int(self.params['num_design_params'])
        self.learning_rate = self.params['learning_rate']
        self.loss_func = self.params['objective_function']
        self.io_mode = self.params['io_mode']
        self.fold_idx = fold_idx
        self.arch, self.linear = None, None
        
        # regular LSTM or ConvLSTM
        if self.params['arch'] == 1:
            self.name = 'lstm'
            self.arch_params = self.params['lstm']
            self.create_architecture()
        elif self.params['arch'] == 2:
            self.name = 'conv_lstm'
            self.arch_params = self.params['conv_lstm']
            self.create_architecture()
        else:
            raise ValueError("Model architecture not recognized")
        
        self.save_hyperparameters()
        
        # Store necessary lists for tracking metrics per fold
        self.test_results = {'train': {'nf_pred': [], 'nf_truth': []},
                             'valid': {'nf_pred': [], 'nf_truth': []}}
        
        # Initialize metrics
        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        
    # setup
    def create_architecture(self):
        seq_len = self.params['seq_len']
        
        if self.name == 'lstm':
            self.arch = torch.nn.LSTM(input_size=self.arch_params['i_dims'],
                                      hidden_size=self.arch_params['h_dims'],
                                      num_layers=self.arch_params['num_layers'],
                                      batch_first=True)
            
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(self.arch_params['h_dims'], self.arch_params['i_dims']),
                torch.nn.Tanh())
            
        else: # ConvLSTM
            kernel_size = self.arch_params['kernel_size']
            padding = self.arch_params['padding']
            out_channels = self.arch_params['out_channels']
            num_layers = self.arch_params['num_layers']
            
            if self.arch_params['use_ae'] == True: # setup autoencoder
                # Calculate size after each conv layer
                temp_spatial = self.arch_params['spatial']
                for _ in range(len(self.arch_params['encoder_channels']) - 1):
                    # mimicking the downsampling to determine the reduced spatial size
                    # (spatial + 2*padding - kernel_size) // stride + 1
                    temp_spatial = ((temp_spatial + 2*1 - 3) // 2) + 1
                reduced_spatial = temp_spatial
                
                # Encoder: downsampling
                self.encoder = Encoder(
                    channels=self.arch_params['encoder_channels'],
                    spatial_size=self.arch_params['spatial']
                )
                
                # Decoder: upsampling
                self.decoder = Decoder(
                    channels=self.arch_params['decoder_channels'],
                    spatial_size=self.arch_params['spatial']
                )
                
                # different since representation space
                in_channels = self.arch_params['encoder_channels'][-1]
                spatial = reduced_spatial
                
            else: # in stays vanilla, full spatial
                in_channels = self.arch_params['in_channels']
                spatial = self.arch_params['spatial'] # 166
            
            # Create single ConvLSTM layer
            self.arch = ConvLSTM(
                in_channels=in_channels,
                out_channels=out_channels,
                seq_len=seq_len,
                kernel_size=kernel_size,
                padding=padding,
                frame_size=(spatial, spatial)
            )
            
            # conv reduction + activation to arrive back at real/imag
            self.linear = nn.Sequential(
                nn.Conv2d(out_channels, 2, kernel_size=1),
                nn.Tanh(),
            )
                
    def configure_optimizers(self):
        # Optimization routine
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # LR scheduler setup
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.params['num_epochs'])
        lr_scheduler_config = {"scheduler": lr_scheduler,
                               "interval": "epoch", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def compute_loss(self, preds, labels, choice='mse'):
        if choice == 'mse':
            # Mean Squared Error
            preds = preds.to(torch.float32).contiguous()
            labels = labels.to(torch.float32).contiguous()
            fn = torch.nn.MSELoss()
            loss = fn(preds, labels)
            
        elif choice == 'emd':
            # ignoring emd for now
            raise NotImplementedError("Earth Mover's Distance not implemented!")
            '''# Earth Mover's Distance / Sinkhorn
            preds = preds.to(torch.float64).contiguous()
            labels = labels.to(torch.float64).contiguous()
            fn = SamplesLoss("sinkhorn", p=1, blur=0.05)
            loss = fn(preds, labels)
            loss = torch.mean(loss)  # Aggregating the loss'''
            
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
    
    def objective(self, preds, labels):
        loss = self.compute_loss(preds, labels, choice=self.loss_func)
        
        return {"loss": loss}
    
    def forward(self, x, meta=None):
        
        # Forward Pass: LSTM
        if self.name == 'lstm':
            batch, seq_len, input = x.size()
            if meta is None: # Init hidden and cell states if not provided
                h = torch.zeros(self.arch_params['num_layers'], 
                                batch, self.arch_params['h_dims']).to(x.device)
                c = torch.zeros(self.arch_params['num_layers'], 
                                batch, self.arch_params['h_dims']).to(x.device)
                meta = (h, c)
            
            predictions = [] # to store each successive pred as we pass through
            
            if self.io_mode == 'one_to_many':
                # first timestep: use our single slice input
                lstm_out, meta = self.arch(x, meta)
                pred = self.linear(lstm_out)
                predictions.append(pred) # t+1 (or + delta split)
                
                # remaining t: no input, we pass 0's as dummy vals
                for _ in range(self.params['seq_len'] - 1):
                    dummy_input = torch.zeros_like(x)
                    lstm_out, meta = self.arch(dummy_input, meta)
                    pred = self.linear(lstm_out)
                    predictions.append(pred)
                
            elif self.io_mode == 'many_to_many':
                # Encoder phase: process all input t's
                for t in range(x.size(1)):
                    lstm_out, meta = self.arch(x[:, t:t+1, :])
                    
                # Decoder phase: generate preds with dummy inputs
                for _ in range(self.params['seq_len']):
                    dummy_input = torch.zeros_like(x[:, 0:1, :]) # the shape of a single t
                    lstm_out, meta = self.arch(dummy_input, meta)
                    pred = self.linear(lstm_out)
                    predictions.append(pred)
                    
            else:
                # other io modes not currently implemented
                return NotImplementedError(f'Recurrent input-output mode "{self.io_mode}" is not implemented.')
                
            # Stack predictions along sequence dimension
            predictions = torch.cat(predictions, dim=1)
            return predictions, meta
        
        # Forward Pass: ConvLSTM
        elif self.name == 'conv_lstm':
            batch, seq_len, r_i, xdim, ydim = x.size()
            x = x.view(batch, seq_len, self.arch_params['in_channels'], 
                       self.arch_params['spatial'], self.arch_params['spatial'])

            if self.arch_params['use_ae'] == True:
                # process sequence through encoder
                encoded_sequence = []
                for t in range(seq_len):
                    # encode each t
                    encoded = self.encoder(x[:, t]) # [batch, latent_dim]
                    encoded_sequence.append(encoded)
                # stack to get [batch, seq_len, latent_dim]
                encoded_sequence = torch.stack(encoded_sequence, dim=1)
                
                # process through LSTM in latent space
                lstm_out, meta = self.arch(encoded_sequence, meta, mode=self.io_mode)
                
                # decode outputted sequence
                decoded_sequence = []
                for t in range(lstm_out.size(1)):
                    # decode each t
                    decoded = self.decoder(lstm_out[:, t])
                    decoded_sequence.append(decoded)
                preds = torch.stack(decoded_sequence, dim=1) 
            
            else: # no AE
                # invoke for specified mode (i.e. many_to_many)
                lstm_out, meta = self.arch(x, meta, mode=self.io_mode)
                # reshape for conv
                b, s, ch, he, w = lstm_out.size()
                lstm_out = lstm_out.view(b * s, ch, he, w)
                # apply conv + tanh
                preds = self.linear(lstm_out)
                # reshape back
                preds = preds.view(b, s, 2, he, w)
            
        else:
            return NotImplementedError(f"Recurrent architecture '{self.name}' is not currently implemented.")
                
        return preds, meta
    
    def shared_step(self, batch, batch_idx):
        samples, labels = batch
        
        # extract sizes - input sequence could be len=1 if 12M
        batch_size, input_seq_len, r_i, xdim, ydim = samples.size()
        
        if self.name == 'lstm':
            # flatten spatial and r/i dims - (batch_size, seq_len, input_size)
            samples = samples.view(batch_size, input_seq_len, -1)
            labels = labels.view(batch_size, self.params['seq_len'], -1)
        
        # Forward pass
        preds, _ = self.forward(samples)
        
        # Compute loss
        loss_dict = self.objective(preds, labels)
        loss = loss_dict['loss']
        
        # reshape preds for metrics
        if self.io_mode == "one_to_many":
            preds = preds.view(batch_size, self.params['seq_len'], r_i, xdim, ydim)
        elif self.io_mode == "many_to_many":
            preds = preds.view(batch_size, self.params['seq_len'], r_i, xdim, ydim)
        else:
            # other modes not implemented
            raise NotImplementedError
        
        return loss, preds
    
    def training_step(self, batch, batch_idx):
        loss, preds = self.shared_step(batch, batch_idx)
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'loss': loss, 'output': preds, 'target': batch}
    
    def validation_step(self, batch, batch_idx):
        loss, preds = self.shared_step(batch, batch_idx)
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'loss': loss, 'output': preds, 'target': batch}
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds = self.shared_step(batch, batch_idx)
        self.organize_testing(preds, batch, batch_idx, dataloader_idx)
        
    def organize_testing(self, preds, batch, batch_idx, dataloader_idx=0):
        samples, labels = batch
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # Determine the mode based on dataloader_idx
        if dataloader_idx == 0:
            mode = 'valid'
        elif dataloader_idx == 1:
            mode = 'train'
        else:
            raise ValueError(f"Invalid dataloader index: {dataloader_idx}")
        
        # Append predictions and truths
        self.test_results[mode]['nf_pred'].append(preds_np)
        self.test_results[mode]['nf_truth'].append(labels_np)

    def on_test_end(self):
        for mode in ['train', 'valid']:
            if self.test_results[mode]['nf_pred']:
                self.test_results[mode]['nf_pred'] = np.concatenate(self.test_results[mode]['nf_pred'], axis=0)
                self.test_results[mode]['nf_truth'] = np.concatenate(self.test_results[mode]['nf_truth'], axis=0)
                
                # Handle fold index
                fold_suffix = f"_fold{self.fold_idx+1}" if self.fold_idx is not None else ""
                
                # Log or save results
                name = f"results{fold_suffix}"
                self.logger.experiment.log_results(
                    results=self.test_results[mode],
                    epoch=None,
                    mode=mode,
                    name=name
                )
            else:
                print(f"No test results for mode: {mode}")