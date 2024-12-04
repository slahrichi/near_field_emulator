import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

class Encoder(nn.Module):
    """Encodes E to a representation space for input to RNN"""
    def __init__(self, channels, params):
        super().__init__()
        
        self.mode = params['modes']
        self.layers = nn.ModuleList()
        current_size = params['spatial']
        current_channels = channels[0]
        
        if self.mode == 'svd':
            # Use SVDEmbedding
            self.svd_embedding = SVDEmbedding(
                in_channels=channels[0],
                out_channels=channels[1],
                spatial_size=params['spatial']
            )
            current_channels = channels[1]  # Update current_channels after SVDEmbedding
            # Start building layers from index 1 since channels[0] and channels[1] are used in SVDEmbedding
            for i in range(1, len(channels) - 1):
                self.layers.extend([
                    nn.Conv2d(channels[i], channels[i+1],
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.LeakyReLU(0.2)
                ])
                current_size = current_size // 2
        else:
            # Standard downsampling approach
            for i in range(len(channels) - 1):
                self.layers.extend([
                    nn.Conv2d(channels[i], channels[i+1], 
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.LeakyReLU(0.2)
                ])
                current_size = current_size // 2
        
        # downsampling layers
        for i in range(len(channels) - 1):
            self.layers.extend([
                nn.Conv2d(channels[i], channels[i+1], 
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.LeakyReLU(0.2)
            ])
            current_size = current_size // 2
            
        self.final_size = current_size
        self.final_channels = channels[-1]
        
    def forward(self, x):
        # x shape [batch, channels, xdim, ydim]
        if self.mode == 'svd':
            x = self.svd_embedding(x)
        for layer in self.layers:
            x = layer(x)
        # Output: [batch, final_channels, reduced_spatial, reduced_spatial]
        return x
    
class Decoder(nn.Module):
    """Decodes latent representation back to E"""
    def __init__(self, channels, params):
        super().__init__()
        
        self.channels = channels
        self.initial_size = params['spatial'] // (2 ** (len(channels) - 1))
        self.layers = nn.ModuleList()
        
        # transposed convolution upsampling
        for i in range(len(channels) - 1):
            self.layers.extend([
                nn.ConvTranspose2d(channels[i], channels[i+1],
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.LeakyReLU(0.2)
            ])
            
    def forward(self, x):
        # x shape: [batch, input_channels, reduced_spatial, reduced_spatial]        
        for layer in self.layers[:-1]: # all but the last
            x = layer(x)
        x = self.layers[-1](x) # last layer w/o activation
        if x.shape[-1] != 166:
            x = x[:, :, :166, :166]  # Crop to exact size
        return x # [batch, 2, 166, 166]
    
class SVDEmbedding(nn.Module):
    """For use with the encoder, dim reductions of E with SVD"""    
    def __init__(self, in_channels, out_channels, spatial_size):
        super(SVDEmbedding, self).__init__()
        self.input_channels = in_channels
        self.spatial_size = spatial_size
        self.out_channels = out_channels
        self.flattened_size = in_channels * spatial_size * spatial_size

        # Define the SVD components
        self.U = nn.Linear(self.flattened_size, self.out_channels, bias=False)
        self.S = nn.Parameter(torch.ones(self.out_channels))
        self.VT = nn.Linear(self.out_channels, self.flattened_size, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        
        x_flat = x.view(batch_size, -1)  # flatten input: [batch_size, flattened_size]
        u = self.U(x_flat)  # [batch_size, out_channels]
        s = torch.diag(self.S)  # apply singular vals: [out_channels, out_channels]
        vt = self.VT.weight  # [flattened_size, out_channels]
        x_svd = u @ s @ vt.T  # reconstruct: [batch_size, flattened_size]
        
        # Reshape back to original spatial dimensions
        x_svd = x_svd.view(batch_size, self.input_channels, self.spatial_size, self.spatial_size)
        return x_svd
    
class Autoencoder(LightningModule):
    def __init__(self, params_model, fold_idx=None):
        super().__init__()
        
        self.params = params_model
        self.encoder_channels = self.params['autoencoder']['encoder_channels']
        self.decoder_channels = self.params['autoencoder']['decoder_channels']
        self.spatial_size = self.params['autoencoder']['spatial']
        self.encoder = Encoder(self.encoder_channels, params=self.params['autoencoder'])
        self.decoder = Decoder(self.decoder_channels, params=self.params['autoencoder'])
        self.learning_rate = self.params['learning_rate']
        self.lr_scheduler = self.params['lr_scheduler']
        self.fold_idx = fold_idx

        self.test_results = {'train': {'nf_pred': [], 'nf_truth': []},
                             'valid': {'nf_pred': [], 'nf_truth': []}}
        
        # Initialize metrics
        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        #self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        #self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        
        self.save_hyperparameters()
        
    def state_dict(self, *args, **kwargs):
        """override state dict to organize things better"""
        state_dict = super().state_dict(*args, **kwargs)
        
        # create nested state dict to separate encoder and decoder
        nested_state_dict = {}
        
        # organize
        for key, value in state_dict.items():
            if key.startswith('encoder'):
                nested_state_dict[key] = value
            elif key.startswith('decoder'):
                nested_state_dict[key] = value
            
        return nested_state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to handle both nested and flat structures  
           Handles older pretrained models that weren't organized with above override"""
        # Check if we have a nested or flat structure
        if 'encoder' in state_dict and 'decoder' in state_dict:
            # Nested structure - flatten it
            flat_state_dict = {}
            for module in ['encoder', 'decoder']:
                for key, value in state_dict[module].items():
                    flat_state_dict[f'{module}.{key}'] = value
            state_dict = flat_state_dict
            
        return super().load_state_dict(state_dict, strict=strict)
        
    def forward(self, x):
        # [batch, 2, 166, 166] -> x
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def shared_step(self, batch, stage="train"):
        x, _ = batch
        if len(x.shape) == 5:
            x = x[:, 0] # [batch, 2, 166, 166]
            
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        
        # calculate addl metrics
        with torch.no_grad():
            psnr = self.train_psnr(x_hat, x) if stage == 'train' else self.val_psnr(x_hat, x)
            #ssim = self.train_ssim(x_hat.unsqueeze(1), x.unsqueeze(1)) if stage == 'train' else self.val_ssim(x_hat.unsqueeze(1), x.unsqueeze(1))
            
        return {
            'loss': loss,
            'psnr': psnr,
            #'ssim': ssim,
            'recon': x_hat
        }

    
    def training_step(self, batch, batch_idx):
        results = self.shared_step(batch, 'train')
        
        # Log all metrics
        self.log('train_loss', results['loss'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_psnr', results['psnr'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        #self.log('train_ssim', results['ssim'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        return results
    
    def validation_step(self, batch, batch_idx):
        results = self.shared_step(batch, 'valid')
        
        # Log all metrics
        self.log('val_loss', results['loss'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_psnr', results['psnr'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        #self.log('val_ssim', results['ssim'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        return results
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # LR scheduler setup
        if self.lr_scheduler == 'CosineAnnealingLR':
            lr_scheduler = CosineAnnealingLR(optimizer, 
                                             T_max=self.params['num_epochs'] * 0.25,
                                             eta_min=1e-6)
        elif self.lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(optimizer, 
                                             mode='min', 
                                             factor=0.5, 
                                             patience=3, 
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
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        results = self.shared_step(batch, 'test')
        self.organize_testing(results['recon'], batch, batch_idx, dataloader_idx)
        
    def organize_testing(self, preds, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        preds = preds.detach().cpu().numpy()
        labels = x.detach().cpu().numpy()
        
        # Determine the mode based on dataloader_idx
        if dataloader_idx == 0:
            mode = 'valid'
        elif dataloader_idx == 1:
            mode = 'train'
        else:
            raise ValueError(f"Invalid dataloader index: {dataloader_idx}")
        
        # Append predictions and truths
        self.test_results[mode]['nf_pred'].append(preds)
        self.test_results[mode]['nf_truth'].append(labels)

    def on_train_epoch_end(self):
        # Get current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=True)

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