import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np

class Encoder(nn.Module):
    """Encodes E to a representation space for input to RNN"""
    def __init__(self, channels, spatial_size):
        super().__init__()
        
        self.layers = nn.ModuleList()
        current_size = spatial_size
        
        # downsampling layers
        for i in range(len(channels) - 1):
            self.layers.extend([
                nn.Conv2d(channels[i], channels[i+1], 
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU()
            ])
            current_size = current_size // 2
            
        self.final_size = current_size
        self.final_channels = channels[-1]
        
    def forward(self, x):
        # x shape [batch, channels, xdim, ydim]
        for layer in self.layers:
            x = layer(x)
        # Output: [batch, final_channels, reduced_spatial, reduced_spatial]
        return x
    
class Decoder(nn.Module):
    """Decodes latent representation back to E"""
    def __init__(self, channels, spatial_size):
        super().__init__()
        
        self.channels = channels
        self.initial_size = spatial_size // (2 ** (len(channels) - 1))
        self.layers = nn.ModuleList()
        
        # transposed convolution upsampling
        for i in range(len(channels) - 1):
            self.layers.extend([
                nn.ConvTranspose2d(channels[i], channels[i+1],
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU()
            ])
            
    def forward(self, x):
        # x shape: [batch, input_channels, reduced_spatial, reduced_spatial]        
        for layer in self.layers[:-1]: # all but the last
            x = layer(x)
        x = self.layers[-1](x) # last layer w/o activation
        if x.shape[-1] != 166:
            x = x[:, :, :166, :166]  # Crop to exact size
        return x # [batch, 2, 166, 166]
    
class Autoencoder(LightningModule):
    def __init__(self, params_model, fold_idx=None):
        super().__init__()
        
        self.params = params_model
        self.encoder_channels = self.params['conv_lstm']['encoder_channels']
        self.decoder_channels = self.params['conv_lstm']['decoder_channels']
        self.spatial_size = self.params['conv_lstm']['spatial']
        self.encoder = Encoder(self.encoder_channels, self.spatial_size)
        self.decoder = Decoder(self.decoder_channels, self.spatial_size)
        self.learning_rate = self.params['learning_rate']
        self.fold_idx = fold_idx

        self.test_results = {'train': {'nf_pred': [], 'nf_truth': []},
                             'valid': {'nf_pred': [], 'nf_truth': []}}
        
    def forward(self, x):
        # [batch, 2, 166, 166] -> x
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def training_step(self, batch, batch_idx):
        x, _ = batch # ignoring labels during pretraining
        if len(x.shape) == 5:
            x = x[:, 0] # [batch, 2, 166, 166]
            
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        if len(x.shape) == 5:
            x = x[:, 0]
        x_hat = self(x)
        val_loss = F.mse_loss(x_hat, x)
        self.log('val_loss', val_loss)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        if len(x.shape) == 5:
            x = x[:, 0]
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log('test_loss', loss)
        
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