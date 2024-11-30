#--------------------------------
# Import: Basic Python Libraries
#--------------------------------
import itertools
import os
import sys
import torch
import logging
import numpy as np
from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, Subset
import pickle
import torch

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
sys.path.append('../')
from core import curvature
from utils import mapping

# debugging
#logging.basicConfig(level=logging.DEBUG)

class NF_Datamodule(LightningDataModule):
    def __init__(self, params, transform = None):
        super().__init__() 
        logging.debug("datamodule.py - Initializing NF_DataModule")

        self.params = params.copy()
        logging.debug("datamodule.py - Setting params to {}".format(self.params))
        self.experiment = params['experiment']
        self.arch = params['arch']
        self.n_cpus = self.params['n_cpus']
        self.seed = self.params['seed']
        self.n_folds = self.params['n_folds']
        self.mlp_strategy = self.params['mlp_strategy']
        self.patch_size = self.params['patch_size']
        self.path_data = self.params['path_data']
        self.path_root = self.params['path_root']
        self.path_data = os.path.join(self.path_root,self.path_data)
        logging.debug("datamodule.py - Setting path_data to {}".format(self.path_data))
        
        self.batch_size = self.params['batch_size']
        self.transform = transform #TODO
        self.dataset = None
        self.train = None
        self.valid = None
        self.test = None

        self.initialize_cpus(self.n_cpus)

    def initialize_cpus(self, n_cpus):
        # Ensure the number of CPUs doesn't exceed the system's capacity
        if n_cpus > os.cpu_count():
            n_cpus = 1
        self.n_cpus = n_cpus 
        logging.debug("NF_DataModule | Setting CPUS to {}".format(self.n_cpus))

    def prepare_data(self):
        # if necessary, preprocessing steps could be built in right here
        #preprocess_data.preprocess_data(path = os.path.join(self.path_data, 'raw'))
        pass

    def setup(self, stage: Optional[str] = None):
        # load the full dataset
        if self.experiment == 1: # autoencoder pretraining
            datafile = os.path.join(self.path_data, 'dataset.pt')
            self.dataset = format_ae_data(datafile, self.params)
        else:
            if self.arch == 0: # MLP
                data = torch.load(os.path.join(self.path_data, 'dataset_nobuffer.pt'))
                if self.params['interpolate_fields']: # interpolate fields to lower resolution
                    data = interpolate_fields(data)
                self.dataset = WaveMLP_Dataset(data, self.transform, self.mlp_strategy, self.patch_size)
            elif self.arch == 1 or self.arch == 2: # LSTM
                datafile = os.path.join(self.path_data, 'dataset.pt')
                self.dataset = format_temporal_data(datafile, self.params)
            
    def setup_fold(self, train_idx, val_idx):
        # create subsets for the current fold
        self.train = Subset(self.dataset, train_idx)
        self.valid = Subset(self.dataset, val_idx)

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.n_cpus,
                          persistent_workers=True,
                          shuffle=True
                        )

    def val_dataloader(self):
        return DataLoader(self.valid,
                          batch_size=self.batch_size,
                          num_workers=self.n_cpus, 
                          shuffle=False,
                          persistent_workers=True
                        )

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.n_cpus,
                          shuffle=False
                        )

class WaveMLP_Dataset(Dataset):
    def __init__(self, data, transform, approach=0, patch_size=1):
        logging.debug("datamodule.py - Initializing WaveMLP_Dataset")
        self.transform = transform
        logging.debug("NF_Dataset | Setting transform to {}".format(self.transform))
        self.approach = approach
        self.patch_size = patch_size
        self.radii = data['radii']
        self.phases = data['phases']
        self.derivatives = data['derivatives']
        # focus on 1550 wavelength in y for now
        temp_nf_1550 = data['all_near_fields']['near_fields_1550']
        temp_nf_1550 = torch.stack(temp_nf_1550, dim=0) # stack all sample tensors
        temp_nf_1550 = temp_nf_1550.squeeze(1) # remove redundant dimension
        temp_nf_1550 = temp_nf_1550[:, 1, :, :, :] # [num_samples, mag/phase, 166, 166]
        # convert to cartesian coords
        mag, phase = mapping.polar_to_cartesian(temp_nf_1550[:, 0, :, :], temp_nf_1550[:, 1, :, :])
        mag = mag.unsqueeze(1)
        phase = phase.unsqueeze(1)
        self.near_fields = torch.cat((mag, phase), dim=1) # [num_samples, r/i, 166, 166]
        
        # distributed subset approach
        if self.approach == 2:
            self.distributed_indices = self.get_distributed_indices()
            
    def get_distributed_indices(self):
        if self.patch_size == 1: # center the single pixel on the middle
            middle_index = 165 // 2
            return np.array([[middle_index, middle_index]])
        else: # generate patch_size evenly distributed indices
            x = np.linspace(0, 165, self.patch_size).astype(int)
            y = np.linspace(0, 165, self.patch_size).astype(int)
            return list(itertools.product(x, y))

    def __len__(self):
        return len(self.near_fields)

    def __getitem__(self, idx):
        near_field = self.near_fields[idx]
        radius = self.radii[idx].float()
        
        if self.approach == 2:
            # selecting patch_size evenly distributed pixels
            x_indices, y_indices = zip(*self.distributed_indices)
            logging.debug(f"WaveMLP_Dataset | x_indices: {x_indices}, y_indices: {y_indices}")
            near_field = near_field[:, x_indices, y_indices]
            near_field = near_field.reshape(2, self.patch_size, self.patch_size)
        if self.transform:   
            near_field = self.transform(near_field)
        
        logging.debug(f"WaveMLP_Dataset | near_field shape: {near_field.shape}")
        
        return near_field, radius
    
class WaveLSTM_Dataset(Dataset):
    def __init__(self, samples, labels):
        logging.debug("datamodule.py - Initializing WaveLSTM_Dataset")
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return len(self.samples)
        
def select_data(params):
    return NF_Datamodule(params)

#--------------------------------
# Initialize: Format data
#--------------------------------

# for saving preprocessed data into a single pt. file (LSTM/RNN)
def load_pickle_data(train_path, valid_path, save_path, arch='mlp'):
    near_fields = []
    phases = []
    derivatives = []
    radii = []
    tag = []
    
    for path in [train_path, valid_path]:
        # keeping track of original split
        normalized_path = os.path.normpath(path)
        parent_dir = os.path.basename(normalized_path)
        is_train = parent_dir == 'train'
        current_tag = 1 if is_train else 0
        
        for current_file in os.listdir(path): # loop through pickle files
            if current_file.endswith(".pkl"):
                current_file_path = os.path.join(path, current_file)
                tag.append(current_tag) # train or valid sample
                
                with open(current_file_path, "rb") as f:
                    data = pickle.load(f)
                    
                    if arch=='mlp':
                        # extracting final slice from meta_atom_rnn data (1550 wl)
                        near_field_sample = data['data'][:, :, :, -1].float()  # [2, 166, 166]
                    elif arch=='lstm':
                        # all slices
                        near_field_sample = data['data'].float()  # [2, 166, 166, 63]
                    else:
                        raise ValueError("Invalid architecture")
                    
                    # append near field and phase data
                    near_fields.append(near_field_sample)
                    phases.append(data['LPA phases'])
                    
                    # per phases, calculate derivatives and append
                    der = curvature.get_der_train(data['LPA phases'].view(1, 3, 3))
                    derivatives.append(der)
                    
                    # per phase, compute radii and store
                    temp_radii = torch.from_numpy(mapping.phase_to_radii(data['LPA phases']))
                    radii.append(temp_radii)
    
    # convert to tensors
    near_fields_tensor = torch.stack([torch.tensor(f) for f in near_fields], dim=0)  # [num_samples, 2, 166, 166, 63]
    phases_tensor = torch.stack([torch.tensor(p) for p in phases], dim=0)  # [num_samples, 9]
    derivatives_tensor = torch.stack([torch.tensor(d) for d in derivatives], dim=0)  # [num_samples, 3, 3]
    radii_tensor = torch.stack([torch.tensor(r) for r in radii], dim=0)  # [num_samples, 9]  
    tag_tensor = torch.tensor(tag) # [num_samples] 1 for train 0 for valid
    
    logging.debug(f"Near fields tensor size: {near_fields_tensor.shape}")
    logging.debug(f"Memory usage: {near_fields_tensor.element_size() * near_fields_tensor.nelement() / 1024**3:.2f} GB")
    
    torch.save({'near_fields': near_fields_tensor, 
                'phases': phases_tensor, 
                'derivatives': derivatives_tensor,
                'radii': radii_tensor,
                'tag': tag_tensor}, save_path)
    print(f"Data saved to {save_path}")
    
def format_temporal_data(datafile, config, order=(-1, 0, 1, 2)):
    """Formats the preprocessed data file into the correct setup  
    and order for the LSTM model.

    Args:
        data (str): path to the file containing the preprocessed data
        seq_len (int): length of the sequence to be used
        order (tuple, optional): order of the sequence to be used. Defaults to (-1, 0, 1, 2).
        
    Returns:
        dataset (WaveLSTM_Dataset): formatted dataset
    """
    all_samples, all_labels = [], []
    data = torch.load(datafile)
    
    # [100, 2, 166, 166, 63] --> access each of the 100 datapoints
    for i in range(data['near_fields'].shape[0]):
        full_sequence = data['near_fields'][i] # [2, xdim, ydim, total_slices]
        total = full_sequence.shape[-1] # all time slices
        
        if config['spacing_mode'] == 'distributed':
            if config['io_mode'] == 'one_to_many':
                # calculate seq_len+1 evenly spaced indices
                indices = np.linspace(1, total-1, config['seq_len']+1)
                distributed_block = full_sequence[:, :, :, indices]
                # the sample is the first one, labels are the rest
                sample = distributed_block[:, :, :, :1]  # [2, xdim, ydim, 1]
                label = distributed_block[:, :, :, 1:]  # [2, xdim, ydim, seq_len]
                
            elif config['io_mode'] == 'many_to_many':
                # input is the first seq_len evenly spaced indices
                input_indices = np.linspace(1, total//2, config['seq_len'])
                sample = full_sequence[:, :, :, input_indices]
                # output is the next seq_len evenly spaced indices
                output_indices = np.linspace(total//2 + 1, total-1, config['seq_len'])
                label = full_sequence[:, :, :, output_indices]
                
            else:
                # many to one, one to one not implemented
                raise NotImplementedError(f'Specified recurrent input-output mode is not implemented.')
            
            # rearrange dims and add to lists
            sample = sample.permute(order) # [1, 2, xdim, ydim]
            label = label.permute(order) # [seq_len, 2, xdim, ydim]
            all_samples.append(sample)
            all_labels.append(label)
            
        # 
        elif config['spacing_mode'] == 'sequential':
            if config['io_mode'] == 'one_to_many':
                #for t in range(0, total, config['seq_len']+1): note: this raise the total number of sample/label pairs
                t = 0
                # check if there are enough timesteps for a full block
                if t + config['seq_len'] < total:
                    block = full_sequence[:, :, :, t:t+config['seq_len'] + 1]
                    # ex: sample -> t=0 , label -> t=1, t=2, t=3 (if seq_len were 3)
                    sample = block[:, :, :, :1]
                    label = block[:, :, :, 1:]
                    sample = sample.permute(order)
                    label = label.permute(order)
                    all_samples.append(sample)
                    all_labels.append(label)
                        
            elif config['io_mode'] == 'many_to_many':
                # true many to many
                sample = full_sequence[:, :, :, :config['seq_len']]
                label = full_sequence[:, :, :, 1:config['seq_len']+1]
                sample = sample.permute(order)
                label = label.permute(order)
                all_samples.append(sample)
                all_labels.append(label)
                
                # this is our 'encoder-decoder' mode - not really realistic here
                '''step_size = 2 * config['seq_len']
                
                #for t in range(0, total, step_size):
                t = 0
                # check if there's enough
                if t + step_size <= total:
                    # input is first seq_len steps in the block
                    sample = full_sequence[:, :, :, t:t+config['seq_len']]
                    # output is next seq_len steps
                    label = full_sequence[:, :, :, t+config['seq_len']:t+step_size]
                    sample = sample.permute(order)
                    label = label.permute(order)
                    all_samples.append(sample)
                    all_labels.append(label)'''
                        
            else:
                raise NotImplementedError(f'Specified recurrent input-output mode is not implemented.')
        
        else:
            # no other spacing modes are implemented
            raise NotImplementedError(f'Specified recurrent dataloading configugration is not implemented.')
        
    return WaveLSTM_Dataset(all_samples, all_labels)

def format_ae_data(datafile, config):
    """Formats the preprocessed data file into the correct setup  
    and order for the autoencoder pretraining.

    Args:
        datafile (str): path to the file containing the preprocessed data
        config (dict): configuration parameters
        
    Returns:
        dataset (WaveLSTM_Dataset): formatted dataset
    """
    all_samples = []
    data = torch.load(datafile)
    
    # 100 samples, 63 slices per sample, 63*100 = 6300 samples/labels to train on
    for i in range(data['near_fields'].shape[0]):
        full_sequence = data['near_fields'][i] # [2, xdim, ydim, total_slices]
        for t in range(full_sequence.shape[-1]):
            sample = full_sequence[:, :, :, t] # [2, xdim, ydim] single sample
            all_samples.append(sample)
            
    # were training on reconstruction, so samples == labels
    return WaveLSTM_Dataset(all_samples, all_samples)
        
def interpolate_fields(data):
    """Interpolates the fields to a lower resolution. Currently supports 2x downsampling.  

    Args:
        data (dict): dictionary containing the near fields, phases, and radii
        
    Returns:
        dataset (WaveLSTM_Dataset): formatted dataset
    """
    near_fields = data['all_near_fields']['near_fields_1550']
    near_fields = torch.stack(near_fields, dim=0)
    # y-component, real component -> [samples, r/i, xdim, ydim]
    real_fields = near_fields[:, :, 1, 0, :, :] 
    imag_fields = near_fields[:, :, 1, 1, :, :]
    # interpolate and combine r/i
    real_fields_interp = torch.nn.functional.interpolate(real_fields, scale_factor=0.5, mode='bilinear')
    imag_fields_interp = torch.nn.functional.interpolate(imag_fields, scale_factor=0.5, mode='bilinear')
    near_fields_new = torch.cat((real_fields_interp, imag_fields_interp), dim=1)
    # create a new list to store interpolated tensors
    near_fields_new_list = []
    for i in range(near_fields.shape[0]):
        # match dimensions accordingly to the original data
        modified = torch.zeros(1, 3, 2, 83, 83)
        modified[0, 1, :, :, :] = near_fields_new[i]
        near_fields_new_list.append(modified)
    
    # update the data    
    data['all_near_fields']['near_fields_1550'] = near_fields_new_list
    
    return data