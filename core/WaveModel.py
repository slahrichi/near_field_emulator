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
import abc

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
#from utils import parameter_manager
from core.ConvLSTM import ConvLSTM
from core.autoencoder import Encoder, Decoder

sys.path.append("../")

class WaveModel(LightningModule, metaclass=abc.ABCMeta):
    """
    Near Field Response Time Series Prediction Model
    Base Abstract Class
    
    Defines a common interface and attributes that all child classes 
    (WaveLSTM, WaveConvLSTM, WaveAELSTM, WaveAEConvLSTM, WaveModeLSTM) must implement.
    """
    def __init__(self, params_model, fold_idx=None):
        super().__init__()
        
        self.params = params_model
        self.fold_idx = fold_idx
        
        # common attributes
        self.learning_rate = self.params['learning_rate']
        self.lr_scheduler = self.params['lr_scheduler']
        self.loss_func = self.params['objective_function']
        self.io_mode = self.params['io_mode']
        self.name = self.params['name']
        
        # store necessary lists for tracking metrics per fold
        self.test_results = {'train': {'nf_pred': [], 'nf_truth': []},
                             'valid': {'nf_pred': [], 'nf_truth': []}}
        
        # initialize metrics
        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        
        # setup architecture
        self.create_architecture()
        
    @abc.abstractmethod
    def create_architecture(self):
        """
        Define model-specific layers and related components here.
        Each subclass must implement this method.
        """
        
    @abc.abstractmethod
    def forward(self, x, meta=None):
        """
        Forward pass of the model.
        Each subclass should implement its own forward logic.
        """
        
    @abc.abstractmethod
    def shared_step(self, batch, batch_idx):
        """
        Method holding model-specific shared logic for training/validation/testing
        Each subclass must implement this method.
        """
        
    @abc.abstractmethod
    def organize_testing(self, preds, batch, batch_idx, dataloader_idx=0):
        """
        Performs general post-processing and loading of testing results for a model.
        Each subclass must implement.
        """

    def compute_loss(self, preds, labels, choice='mse'):
        """
        Compute loss given predictions and labels.
        Subclasses can override if needed, but this base implementation is standard.
        """
        if choice == 'mse':
            # Mean Squared Error
            preds = preds.to(torch.float32).contiguous()
            labels = labels.to(torch.float32).contiguous()
            fn = torch.nn.MSELoss()
            loss = fn(preds, labels)
            
        elif choice == 'emd':
            # ignoring emd for now
            '''# Earth Mover's Distance / Sinkhorn
            preds = preds.to(torch.float64).contiguous()
            labels = labels.to(torch.float64).contiguous()
            fn = SamplesLoss("sinkhorn", p=1, blur=0.05)
            loss = fn(preds, labels)
            loss = torch.mean(loss)  # Aggregating the loss'''
            raise NotImplementedError("Earth Mover's Distance not implemented!")
            
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
        """
        A wrapper method around compute_loss to provide a unified interface.
        """
        return {"loss": self.compute_loss(preds, labels, choice=self.loss_func)}
    
    def configure_optimizers(self):
        """
        Setup optimzier and LR scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # LR scheduler setup - 2 options
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
                                             cooldown=2)
            
        else:
            raise ValueError(f"Unsupported LR scheduler: {self.lr_scheduler}")
        
        lr_scheduler_config = {"scheduler": lr_scheduler,
                               "interval": "epoch", "frequency": 1,
                               "monitor": "val_loss"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def training_step(self, batch, batch_idx):
        """
        Common training step likely shared among all subclasses
        """
        loss, preds = self.shared_step(batch, batch_idx)
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'loss': loss, 'output': preds, 'target': batch}
    
    def validation_step(self, batch, batch_idx):
        """
        Common training step likely shared among all subclasses
        """
        loss, preds = self.shared_step(batch, batch_idx)
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'loss': loss, 'output': preds, 'target': batch}
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Common testing step likely shared among all subclasses
        """
        loss, preds = self.shared_step(batch, batch_idx)
        self.organize_testing(preds, batch, batch_idx, dataloader_idx)
    
    def on_test_end(self):
        """
        After testing, this method compiles results and logs them.
        """
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

class WaveLSTM(WaveModel):
    """Near Field Response Time Series Prediction Model  
    Architecture: LSTM"""
    def __init__(self, params_model, fold_idx=None):
        super().__init__(params_model, fold_idx)
        
        self.arch_params = self.params[self.name] 

    def create_architecture(self):
            i_dims = self.arch_params['i_dims']
                
            self.arch = torch.nn.LSTM(input_size=i_dims,
                                      hidden_size=self.arch_params['h_dims'],
                                      num_layers=self.arch_params['num_layers'],
                                      batch_first=True)
                
            self.linear = torch.nn.Sequential(
                          torch.nn.Linear(self.arch_params['h_dims'], i_dims),
                          torch.nn.Tanh())
        
    def forward(self, x, meta=None):
        batch, seq_len, input = x.size()
        if meta is None: # Init hidden and cell states if not provided
            h, c = self.init_hidden(batch)
            meta = (h, c)          
            
        # flatten spatial and r/i dims - (batch_size, seq_len, input_size)
        x = x.view(batch, seq_len, -1)
        
        predictions = [] # to store each successive pred as we pass through
        
        if self.io_mode == 'one_to_many':
            # first timestep: use our single slice input
            lstm_out, meta = self.arch(x, meta)
            pred = self.linear(lstm_out)
            predictions.append(pred) # t+1 (or + delta split)
            
            # remaining t: no input, we pass 0's as dummy vals
            for t in range(1, self.params['seq_len']):
                dummy_input = torch.zeros_like(x)
                lstm_out, meta = self.arch(dummy_input, meta)
                pred = self.linear(lstm_out)
                predictions.append(pred)
            
        elif self.io_mode == 'many_to_many':
            if self.params['autoreg']:
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
        
    def shared_step(self, batch, batch_idx):
        samples, labels = batch
        
        # extract sizes - input sequence could be len=1 if 12M
        batch_size, input_seq_len, r_i, xdim, ydim = samples.size()
        

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
    
    def init_hidden(self, batch_size):
        h = torch.zeros(self.arch_params['num_layers'], 
                        batch_size, self.arch_params['h_dims']).to(self.device)
        c = torch.zeros(self.arch_params['num_layers'], 
                        batch_size, self.arch_params['h_dims']).to(self.device)
        return (h, c)
        
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
        
        # Append predictions
        self.test_results[mode]['nf_pred'].append(preds_np)
        self.test_results[mode]['nf_truth'].append(labels_np)

        
class WaveConvLSTM(WaveModel):
    """Near Field Response Time Series Prediction Model  
    Architecture: ConvLSTM"""
    def __init__(self, params_model, fold_idx=None):
        super().__init__(params_model, fold_idx)
        
        self.arch_params = self.params[self.name]
        
    def create_architecture(self):
        seq_len = self.params['seq_len']
        kernel_size = self.arch_params['kernel_size']
        padding = self.arch_params['padding']
        out_channels = self.arch_params['out_channels']
        num_layers = self.arch_params['num_layers']
        in_channels = self.arch_params['in_channels']
        spatial = self.arch_params['spatial']
        
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
        
    def forward(self, x, meta=None):
        batch, seq_len, r_i, xdim, ydim = x.size()
        x = x.view(batch, seq_len, self.arch_params['in_channels'], 
                    self.arch_params['spatial'], self.arch_params['spatial'])
        
        # invoke for specified mode (i.e. many_to_many)
        lstm_out, meta = self.arch(x, meta, mode=self.io_mode, 
                                    autoregressive=self.params['autoreg'])
        
        # reshape for conv
        b, s, ch, he, w = lstm_out.size()
        lstm_out = lstm_out.view(b * s, ch, he, w)
        # apply conv + tanh
        preds = self.linear(lstm_out)
        # reshape back
        preds = preds.view(b, s, 2, he, w)

        return preds, meta
        
    def shared_step(self, batch, batch_idx):
        samples, labels = batch
        
        # extract sizes - input sequence could be len=1 if 12M
        batch_size, input_seq_len, r_i, xdim, ydim = samples.size()
        
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
        
        # Append predictions
        self.test_results[mode]['nf_pred'].append(preds_np)
        self.test_results[mode]['nf_truth'].append(labels_np)
        
class WaveAELSTM(WaveModel):
    """Near Field Response Time Series Prediction Model  
    Architecture: LSTM with Autoencoder"""
    def __init__(self, params_model, fold_idx=None):
        super().__init__(params_model, fold_idx)
        
        base_name = self.name.split('-')[-1] # convlstm or lstm
        self.arch_params = {**self.params['autoencoder'], **self.params[base_name]}
        self.encoding_done = False
        
    def create_architecture(self):
        seq_len = self.params['seq_len']
        
        # configure autoencoder to get reduced image
        _, _ = self.configure_ae()
        # flatten representation
        i_dims = self.arch_params['latent_dim']
                
        self.arch = torch.nn.LSTM(input_size=i_dims,
                                    hidden_size=self.arch_params['h_dims'],
                                    num_layers=self.arch_params['num_layers'],
                                    batch_first=True)
        
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(self.arch_params['h_dims'], i_dims),
            torch.nn.Tanh())
        
    def forward(self, x, meta=None):
        self.encoding_done = False
        batch, seq_len, input = x.size()
        x = self.process_ae(x) # encoding step
        if meta is None: # Init hidden and cell states if not provided
            h, c = self.init_hidden(batch)
            meta = (h, c)          
                
        # flatten spatial and r/i dims - (batch_size, seq_len, input_size)
        x = x.view(batch, seq_len, -1)
        
        predictions = [] # to store each successive pred as we pass through
        # prepare output tensor
        output = torch.zeros(batch, self.params['seq_len'],
                            x.shape[2], device=x.device)
        
        if self.io_mode == 'one_to_many':
            # first timestep: use our single slice input
            lstm_out, meta = self.arch(x, meta)
            pred = self.linear(lstm_out)
            output[:, 0] = pred.squeeze(dim=1)
            predictions.append(pred) # t+1 (or + delta split)
            
            # remaining t: no input, we pass 0's as dummy vals
            for t in range(1, self.params['seq_len']):
                dummy_input = torch.zeros_like(x)
                lstm_out, meta = self.arch(dummy_input, meta)
                pred = self.linear(lstm_out)
                output[:, t] = pred.squeeze(dim=1)
                predictions.append(pred)
            
        elif self.io_mode == 'many_to_many':
            if self.params['autoreg']: # testing (probably)
                # Use first timestep
                current_input = x[:, 0]  # Keep seq_len dim with size 1
                current_input = current_input.unsqueeze(1)
                lstm_out, meta = self.arch(current_input, meta)
                pred = self.linear(lstm_out)
                output[:, 0] = pred.squeeze(dim=1)
                predictions.append(pred)
                
                # Generate remaining predictions using previous outputs
                for t in range(1, self.params['seq_len']):
                    # Use previous prediction as input
                    lstm_out, meta = self.arch(pred, meta)
                    pred = self.linear(lstm_out)
                    output[:, t] = pred.squeeze(dim=1)
                    predictions.append(pred)
                    
            else: # teacher forcing
                for t in range(self.params['seq_len']):
                    current_input = x[:, t] # ground truth at t
                    lstm_out, meta = self.arch(current_input, meta)
                    pred = self.linear(lstm_out)
                    output[:, t] = pred.squeeze(dim=1)
                    predictions.append(pred)
                
        else:
            # other io modes not currently implemented
            return NotImplementedError(f'Recurrent input-output mode "{self.io_mode}" is not implemented.')
        
        if self.arch_params['use_decoder'] == True:
            predictions = self.process_ae(x, output) # decoding step
        else: # no decoding
            predictions = output
        # flatten spatial and r/i dims - (batch_size, seq_len, input_size)
        predictions = predictions.view(batch, self.params['seq_len'], -1)
        
        return predictions, meta
        
    def shared_step(self, batch, batch_idx):
        samples, labels = batch
        # extract sizes - input sequence could be len=1 if 12M
        batch_size, input_seq_len, r_i, xdim, ydim = samples.size()
        # flatten spatial and r/i dims - (batch_size, seq_len, input_size)
        samples = samples.view(batch_size, input_seq_len, -1)
        labels = labels.view(batch_size, self.params['seq_len'], -1)
        
        # Forward pass
        preds, _ = self.forward(samples)
        
        # format labels
        if self.arch_params['use_decoder'] == False:
            self.encoding_done = False # reset bc we need to encode again
            labels = self.process_ae(labels) # encode ground truths
            # flatten latent space - (i.e. 512 -> 2, 16, 16)
            if self.arch_params['modes'] == 'linear':
                latent_spatial = np.sqrt(self.arch_params['latent_dim']/2).astype(int)
                xdim, ydim = latent_spatial, latent_spatial
            else:
                raise ValueError(f"Unsupported AE mode for WaveAELSTM: {self.arch_params['mode']}")
            
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
        
        # Append predictions
        self.test_results[mode]['nf_pred'].append(preds_np)
        
        # need to determine if we decoded to match preds and labels
        if self.arch_params['use_decoder'] == False:
            self.encoding_done = False # reset bc we need to encode again
            labels = labels.view(labels.size(0), labels.size(1), -1)
            labels = self.process_ae(labels) # encode ground truths
            # flatten latent space - (i.e. 512 -> 2, 16, 16)
            # convert latent_dim to int
            if self.arch_params['modes'] == 'linear':
                latent_spatial = np.sqrt(self.arch_params['latent_dim']/2).astype(int)
                xdim, ydim = latent_spatial, latent_spatial
            else:
                raise ValueError(f"Unsupported AE mode for WaveAELSTM: {self.arch_params['mode']}")
            labels = labels.view(labels.size(0), labels.size(1), 2, xdim, ydim)
            self.test_results[mode]['nf_truth'].append(labels.detach().cpu().numpy())
        else:
            self.test_results[mode]['nf_truth'].append(labels_np)
            
    def configure_ae(self):       
        # Encoder: downsampling
        self.encoder = Encoder(
            channels=self.arch_params['encoder_channels'],
            params=self.params['autoencoder']
        )
        
        # Decoder: upsampling
        if self.arch_params['use_decoder'] == True:
            self.decoder = Decoder(
                channels=self.arch_params['decoder_channels'],
                params=self.params['autoencoder']
            )
    
        if self.arch_params['pretrained'] == True:
            # load pretrained autoencoder
            dirpath = '/develop/' + self.params['path_pretrained_ae']
            checkpoint = torch.load(dirpath + "model.ckpt")
            encoder_state_dict = {}

            # extract layers for the encoder
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('encoder'):
                    encoder_state_dict[key.replace('encoder.', '')] = value
            
            # load state dict
            self.encoder.load_state_dict(encoder_state_dict)
            
            # freeze ae weights
            if self.arch_params['freeze_weights'] == True:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                    
            # only if we're doing decoding now       
            if self.arch_params['use_decoder'] == True:
                decoder_state_dict = {}
                for key, value in checkpoint['state_dict'].items():
                    if key.startswith('decoder'):
                        decoder_state_dict[key.replace('decoder.', '')] = value    
                self.decoder.load_state_dict(decoder_state_dict)   
                # freeze ae weights
                if self.arch_params['freeze_weights'] == True:
                    for param in self.encoder.parameters():
                        param.requires_grad = False       
                
        # different since representation space
        in_channels = self.arch_params['encoder_channels'][-1]
        spatial = self.arch_params['spatial']
        
        return in_channels, spatial
    
    def process_ae(self, x, lstm_out=None):
        if self.encoding_done == False: # encode the input
            batch, seq_len, input = x.size()
            if input != 55112:
                raise Warning(f"Input flattened size must be 55112, got {input}.")
            if lstm_out is None: # encoder
                # process sequence through encoder
                encoded_sequence = []
                if self.arch_params['freeze_weights'] == True:
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
                self.encoding_done = True
                return encoded_sequence
            
        elif lstm_out is not None: # decoder
            # decode outputted sequence
            decoded_sequence = []
            if self.arch_params['freeze_weights'] == True:
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
            
            return preds
        
        else: # already encoded
            return x
        
class WaveAEConvLSTM(WaveModel):
    """Near Field Response Time Series Prediction Model  
    Architecture: ConvLSTM with Autoencoder"""
    def __init__(self, params_model, fold_idx=None):
        super().__init__(params_model, fold_idx)
        
        base_name = self.name.split('-')[-1] # convlstm or lstm
        self.arch_params = {**self.params['autoencoder'], **self.params[base_name]}
        self.encoding_done = False
        
    def create_architecture(self):
        seq_len = self.params['seq_len']
        kernel_size = self.arch_params['kernel_size']
        padding = self.arch_params['padding']
        out_channels = self.arch_params['out_channels']
        num_layers = self.arch_params['num_layers']
        
        # get the correct sizes for AE
        in_channels, spatial = self.configure_ae()
        
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
            nn.Tanh())
        
    def forward(self, x, meta=None):
        self.encoding_done = False
        try:
            x = self.process_ae(x) # encoding step
        except Warning:
            pass
            
        # invoke for specified mode (i.e. many_to_many)
        lstm_out, meta = self.arch(x, meta, mode=self.io_mode, 
                                    autoregressive=self.params['autoreg'])
        
        if self.arch_params['use_decoder'] == True:
            preds = self.process_ae(x, lstm_out) # decoding step
        else: # no decoding
            preds = lstm_out

        return preds, meta
        
    def shared_step(self, batch, batch_idx):
        samples, labels = batch
        # extract sizes - input sequence could be len=1 if 12M
        batch_size, input_seq_len, r_i, xdim, ydim = samples.size()
        
        # Forward pass
        preds, _ = self.forward(samples)
        
        # encoding the labels
        if self.arch_params['use_decoder'] == False:
            self.encoding_done = False # reset bc we need to encode again
            labels = self.process_ae(labels) # encode ground truths
            if self.arch_params['modes'] == 'conv':
                raise NotImplementedError("need to finish implementing conv encoding for labels")
            else:
                raise ValueError(f"Unsupported mode for WaveAEConvLSTM: {self.arch_params['mode']}")
            
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
        
        # Append predictions
        self.test_results[mode]['nf_pred'].append(preds_np)
        
        # need to determine if we decoded to match preds and labels
        if self.arch_params['use_decoder'] == False:
            self.encoding_done = False # reset bc we need to encode again
            labels = labels.view(labels.size(0), labels.size(1), -1)
            labels = self.process_ae(labels) # encode ground truths
            if self.arch_params['modes'] == 'conv':
                raise NotImplementedError("need to finish implementing WaveAEConvLSTM")
            else:
                raise ValueError(f"Unsupported AE mode for WaveAEConvLSTM: {self.arch_params['mode']}")
            labels = labels.view(labels.size(0), labels.size(1), 2, xdim, ydim)
            self.test_results[mode]['nf_truth'].append(labels.detach().cpu().numpy())
        else:
            self.test_results[mode]['nf_truth'].append(labels_np)
            
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
        if self.arch_params['use_decoder'] == True:
            self.decoder = Decoder(
                channels=self.arch_params['decoder_channels'],
                params=self.params['autoencoder']
            )
    
        if self.arch_params['pretrained'] == True:
            # load pretrained autoencoder
            dirpath = '/develop/' + self.params['path_pretrained_ae']
            checkpoint = torch.load(dirpath + "model.ckpt")
            encoder_state_dict = {}

            # extract layers for the encoder
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('encoder'):
                    encoder_state_dict[key.replace('encoder.', '')] = value
            
            # load state dict
            self.encoder.load_state_dict(encoder_state_dict)
            
            # freeze ae weights
            if self.arch_params['freeze_weights'] == True:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                    
            # only if we're doing decoding now       
            if self.arch_params['use_decoder'] == True:
                decoder_state_dict = {}
                for key, value in checkpoint['state_dict'].items():
                    if key.startswith('decoder'):
                        decoder_state_dict[key.replace('decoder.', '')] = value    
                self.decoder.load_state_dict(decoder_state_dict)   
                # freeze ae weights
                if self.arch_params['freeze_weights'] == True:
                    for param in self.encoder.parameters():
                        param.requires_grad = False       
                
        # different since representation space
        in_channels = self.arch_params['encoder_channels'][-1]
        spatial = reduced_spatial
        
        return in_channels, spatial
    
    def process_ae(self, x, lstm_out=None):
        if self.encoding_done == False: # encode the input
            batch, seq_len, r_i, xdim, ydim = x.size()
            if xdim != 166:
                raise Warning(f"Input spatial size must be 166, got {xdim}.")
            x = x.view(batch, seq_len, 2, 
                self.arch_params['spatial'], self.arch_params['spatial'])
            if lstm_out is None: # encoder
                # process sequence through encoder
                encoded_sequence = []
                if self.arch_params['freeze_weights'] == True:
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
                self.encoding_done = True
                return encoded_sequence
            
        elif lstm_out is not None: # decoder
            # decode outputted sequence
            decoded_sequence = []
            if self.arch_params['freeze_weights'] == True:
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
            
            return preds
        
        else: # already encoded
            return x
        
class WaveModeLSTM(WaveModel):
    """Near Field Response Time Series Prediction Model  
    Architecture: LSTM with Mode Encoder"""
    def __init__(self, params_model, fold_idx=None):
        super().__init__(params_model, fold_idx)
        
    def create_architecture(self):
        pass
        
    def forward(self, x, meta=None):
        pass
        
    def shared_step(self, batch, batch_idx):
        pass
        
    def organize_testing(self, preds, batch, batch_idx, dataloader_idx=0):
        pass