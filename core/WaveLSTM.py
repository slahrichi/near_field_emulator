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

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
#from utils import parameter_manager
from core.ConvLSTM import ConvLSTM
from core.autoencoder import Encoder, Decoder

sys.path.append("../")

class WaveLSTM(LightningModule):
    """Near Field Response Time Series Prediction Model  
    Architecture: LSTM"""
    def __init__(self, params_model, fold_idx=None):
        super().__init__()
        
        self.params = params_model
        self.num_design_params = int(self.params['num_design_params'])
        self.learning_rate = self.params['learning_rate']
        self.lr_scheduler = self.params['lr_scheduler']
        self.loss_func = self.params['objective_function']
        self.io_mode = self.params['io_mode']
        self.fold_idx = fold_idx
        self.linear = None
        self.name = self.params['name'] # which architecture?
        if self.name == 'ae-lstm' or self.name == 'ae-convlstm':
            base_name = self.name.split('-')[-1] # convlstm or lstm
            self.arch_params = {**self.params['autoencoder'], **self.params[base_name]}
        else:
            self.arch_params = self.params[self.name]
        
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
        
        # Vanilla LSTM
        if self.name == 'lstm' or self.name == '-':
            if self.name == '-':
                # configure autoencoder to get reduced image
                in_channels, spatial = self.configure_ae()
                # flatten representation
                i_dims = in_channels * spatial * spatial
            else:
                i_dims = self.arch_params['i_dims']
            
            self.arch = torch.nn.LSTM(input_size=i_dims,
                                      hidden_size=self.arch_params['h_dims'],
                                      num_layers=self.arch_params['num_layers'],
                                      batch_first=True)
            
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(self.arch_params['h_dims'], self.arch_params['i_dims']),
                torch.nn.Tanh())
            
        # Convolutional LSTM
        else:
            kernel_size = self.arch_params['kernel_size']
            padding = self.arch_params['padding']
            out_channels = self.arch_params['out_channels']
            num_layers = self.arch_params['num_layers']
            
            if self.name == 'ae-convlstm':
                in_channels, spatial = self.configure_ae()
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
            
    def configure_ae(self):
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
            params=self.params['autoencoder']
        )
        
        # Decoder: upsampling
        self.decoder = Decoder(
            channels=self.arch_params['decoder_channels'],
            params=self.params['autoencoder']
        )
    
        if self.arch_params['pretrained'] == True:
            # load pretrained autoencoder
            dirpath = '/develop/' + self.params['path_pretrained_ae']
            checkpoint = torch.load(dirpath + "model.ckpt")
            
            # extract respective layers for encoder and decoder
            encoder_state_dict = {}
            decoder_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('encoder'):
                    encoder_state_dict[key.replace('encoder.', '')] = value
                elif key.startswith('decoder'):
                    decoder_state_dict[key.replace('decoder.', '')] = value
            
            # load respective state dicts
            self.encoder.load_state_dict(encoder_state_dict)
            self.decoder.load_state_dict(decoder_state_dict)
            
            # freeze ae weights
            if self.arch_params['freeze_weights'] == True:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                for param in self.decoder.parameters():
                    param.requires_grad = False
    
        # different since representation space
        in_channels = self.arch_params['encoder_channels'][-1]
        spatial = reduced_spatial
        
        return in_channels, spatial
                
    def configure_optimizers(self):
        # Optimization routine
        #trainable_params = [p for p in self.parameters() if p.requires_grad]
        trainable_params = self.parameters()
        optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)
        
        # LR scheduler setup
        if self.lr_scheduler == 'CosineAnnealingLR':
            lr_scheduler = CosineAnnealingLR(optimizer, 
                                             T_max=self.params['num_epochs'],
                                             eta_min=1e-6)
        elif self.lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(optimizer, 
                                             mode='min', 
                                             factor=0.5, 
                                             patience=5, 
                                             min_lr=1e-6, 
                                             threshold=0.001, 
                                             cooldown=2
                                            )
        else:
            raise ValueError(f"Unsupported LR scheduler: {self.lr_scheduler}")
        lr_scheduler_config = {"scheduler": lr_scheduler,
                               "interval": "epoch", "frequency": 1,
                               "monitor": "val_loss"}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def compute_loss(self, preds, labels, choice='mse'):
        # Ensure tensors are properly set up for gradient computation
        #if not preds.requires_grad:
        #    preds = preds.detach().requires_grad_()
        
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
    
    def init_hidden(self, batch_size):
        h = torch.zeros(self.arch_params['num_layers'], 
                        batch_size, self.arch_params['h_dims']).to(self.device)
        c = torch.zeros(self.arch_params['num_layers'], 
                        batch_size, self.arch_params['h_dims']).to(self.device)
        return (h, c)
    
    def forward(self, x, meta=None):
        # autoregressive if we're testing
        #autoreg = not self.training
        autoreg = True # nvm terrible idea to have it on only for testing
        
        # Forward Pass: LSTM
        if self.name == 'lstm':
            batch, seq_len, input = x.size()
            if meta is None: # Init hidden and cell states if not provided
                h, c = self.init_hidden(batch)
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
                if autoreg: # testing (probably)
                    # Use first timestep
                    current_input = x[:, 0]  # Keep seq_len dim with size 1
                    current_input = current_input.unsqueeze(1)
                    lstm_out, meta = self.arch(current_input, meta)
                    pred = self.linear(lstm_out)
                    predictions.append(pred)
                    
                    # Generate remaining predictions using previous outputs
                    for t in range(1, self.params['seq_len']):
                        # Use previous prediction as input
                        lstm_out, meta = self.arch(pred, meta)
                        pred = self.linear(lstm_out)
                        predictions.append(pred)
                else: # teacher forcing
                    for t in range(self.params['seq_len']):
                        current_input = x[:, t] # ground truth at t
                        lstm_out, meta = self.arch(current_input, meta)
                        pred = self.linear(lstm_out)
                        predictions.append(pred)
                    
            else:
                # other io modes not currently implemented
                return NotImplementedError(f'Recurrent input-output mode "{self.io_mode}" is not implemented.')
                
            # Stack predictions along sequence dimension
            predictions = torch.cat(predictions, dim=1)
            return predictions, meta
        
        # Forward Pass: ConvLSTM
        elif self.name == 'convlstm':
            batch, seq_len, r_i, xdim, ydim = x.size()
            x = x.view(batch, seq_len, self.arch_params['in_channels'], 
                       self.arch_params['spatial'], self.arch_params['spatial'])

            if self.arch_params['use_ae'] == True:
                # process sequence through encoder
                encoded_sequence = []
                if self.arch_params['freeze_ae_weights'] == True:
                    with torch.no_grad(): # no grad for pretrained ae
                        for t in range(seq_len):
                            # encode each t
                            encoded = self.encoder(x[:, t]) # [batch, latent_dim]
                            encoded_sequence.append(encoded)
                else: # trainable ae
                    for t in range(seq_len):
                        # encode each t
                        encoded = self.encoder(x[:, t]) # [batch, latent_dim]
                        encoded_sequence.append(encoded)
                # stack to get [batch, seq_len, latent_dim]
                encoded_sequence = torch.stack(encoded_sequence, dim=1)
                
                # process through LSTM in latent space
                lstm_out, meta = self.arch(encoded_sequence, meta, mode=self.io_mode,
                                           autoregressive=autoreg)
                
                # decode outputted sequence
                decoded_sequence = []
                if self.arch_params['freeze_ae_weights'] == True:
                    with torch.no_grad(): # no grad for pretrained ae
                        for t in range(lstm_out.size(1)):
                            # decode each t
                            decoded = self.decoder(lstm_out[:, t])
                            decoded_sequence.append(decoded)
                else: # trainable decoder
                    for t in range(lstm_out.size(1)):
                        # decode each t
                        decoded = self.decoder(lstm_out[:, t])
                        decoded_sequence.append(decoded)
                preds = torch.stack(decoded_sequence, dim=1)
                # ensure predictions can recieve gradients
                #preds = preds.detach().requires_grad_()
                    
            else: # no AE
                # invoke for specified mode (i.e. many_to_many)
                lstm_out, meta = self.arch(x, meta, mode=self.io_mode, 
                                           autoregressive=autoreg)
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
        '''# Only process validation results
        if self.test_results['valid']['nf_pred']:
            # get train results too
            
            self.test_results['valid']['nf_pred'] = np.concatenate(self.test_results['valid']['nf_pred'], axis=0)
            self.test_results['valid']['nf_truth'] = np.concatenate(self.test_results['valid']['nf_truth'], axis=0)
            
            # Handle fold index
            #fold_suffix = f"_fold{self.fold_idx+1}" if self.fold_idx is not None else ""
            #name = f"results{fold_suffix}"
            
            # Log or save results
            self.logger.experiment.log_results(
                results=self.test_results['valid'],
                epoch=None,
                mode='valid',
                name="results"
            )
        else:
            print(f"No test results.")'''
        for mode in ['train', 'valid']:
            if self.test_results[mode]['nf_pred']:
                self.test_results[mode]['nf_pred'] = np.concatenate(self.test_results[mode]['nf_pred'], axis=0)
                self.test_results[mode]['nf_truth'] = np.concatenate(self.test_results[mode]['nf_truth'], axis=0)
                
                # Handle fold index
                #fold_suffix = f"_fold{self.fold_idx+1}" if self.fold_idx is not None else ""
                #name = f"results{fold_suffix}"
                                
                # Log or save results
                self.logger.experiment.log_results(
                    results=self.test_results[mode],
                    epoch=None,
                    mode=mode,
                    name="results"
                )
            else:
                print(f"No test results for mode: {mode}")