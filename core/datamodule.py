#--------------------------------
# Import: Basic Python Libraries
#--------------------------------
import itertools
import os
import sys
import glob #TODO: Needed?
import torch
import logging
import numpy as np #TODO: Needed?
from typing import Optional
#from torchvision import transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, Subset
import pickle
import yaml
import torch
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
sys.path.append('../')
#from core import custom_transforms as ct
#from core import preprocess_data
from core import curvature
from utils import parameter_manager, mapping

class NF_Datamodule(LightningDataModule):
    def __init__(self, params, transform = None):
        super().__init__() 
        logging.debug("datamodule.py - Initializing NF_DataModule")

        self.params = params.copy()
        logging.debug("datamodule.py - Setting params to {}".format(self.params))
        self.arch = params['arch']
        self.n_cpus = self.params['n_cpus']
        self.seed = self.params['seed']
        self.n_folds = self.params['n_folds']
        self.mlp_strategy = self.params['mlp_strategy']

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
        if self.arch == 0: # MLP
            data = torch.load(os.path.join(self.path_data, 'dataset.pt'))
            self.dataset = WaveMLP_Dataset(data, self.transform, self.mlp_strategy)
        elif self.arch == 1 or self.arch == 2: # LSTM
            datafile = os.path.join(self.path_data, 'slices_dataset.pt')
            self.dataset = format_temporal_data(datafile, self.params['seq_len'])
            
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
    def __init__(self, data, transform, approach=0):
        logging.debug("datamodule.py - Initializing WaveMLP_Dataset")
        self.transform = transform
        self.approach = approach
        logging.debug("NF_Dataset | Setting transform to {}".format(self.transform))
        self.radii = data['radii']
        self.phases = data['phases']
        self.derivatives = data['derivatives']
        
        # focus on 1550 wavelength in y for now
        temp_nf_1550 = data['all_near_fields']['near_fields_1550']
        temp_nf_1550 = torch.stack(temp_nf_1550, dim=0) # stack all sample tensors
        temp_nf_1550 = temp_nf_1550.squeeze(1) # remove redundant dimension
        self.near_fields = temp_nf_1550[:, 1, :, :, :] # [num_samples, r/i, 166, 166]
        
        # distributed subset approach
        if self.approach == 2:
            self.distributed_indices = self.get_distributed_indices()
            
    def get_distributed_indices(self):
        # generate 9 evenly distributed indices across the whole spatial output field
        x = np.linspace(0, 165, 3).astype(int)
        y = np.linspace(0, 165, 3).astype(int)
        return list(itertools.product(x, y))

    def __len__(self):
        return len(self.near_fields)

    def __getitem__(self, idx):
        near_field = self.near_fields[idx]
        radius = self.radii[idx].float()
        
        if self.approach == 2:
            # selecting 9 evenly distributed pixels
            x_indices, y_indices = zip(*self.distributed_indices)
            near_field = near_field[:, x_indices, y_indices]
            near_field = near_field.reshape(2, 3, 3)
        if self.transform:   
            near_field = self.transform(near_field)
        
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
    
    for path in [train_path, valid_path]:
            for current_file in os.listdir(path): # loop through pickle files
                if current_file.endswith(".pkl"):
                    current_file_path = os.path.join(path, current_file)
                    
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
    
    torch.save({'near_fields': near_fields_tensor, 
                'phases': phases_tensor, 
                'derivatives': derivatives_tensor,
                'radii': radii_tensor}, save_path)
    print(f"Data saved to {save_path}")
    
def format_temporal_data(datafile, seq_len, order=(-1, 0, 1, 2)):
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
    
    # load the data
    data = torch.load(datafile)
    
    # [100, 2, 166, 166, 63] --> access each of the 100 datapoints
    for i in range(data['near_fields'].shape[0]):
        full_sequence = data['near_fields'][i] # [2, xdim, ydim, total_slices]
        sample = full_sequence[:, :, :, :1] # t=0 slice
        total = full_sequence.shape[-1] # all time slices
        indices = np.linspace(1, total-1, seq_len).astype(int) # indices for sequence blocks
        label = full_sequence[:, :, :, indices] # [2, xdim, ydim, seq_len]
        
        # rearrange dims
        sample = sample.permute(order) # [1, 2, xdim, ydim]
        label = label.permute(order) # [seq_len, 2, xdim, ydim]
        
        all_samples.append(sample)
        all_labels.append(label)
        
    return WaveLSTM_Dataset(all_samples, all_labels)
    
#--------------------------------
# Initialize: Testing
#--------------------------------

if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG)
    seed_everything(1337)
    os.environ['SLURM_JOB_ID'] = '0'
    #plt.style.use(['science'])

    #Load config file   
    params = yaml.load(open('config.yaml'), Loader = yaml.FullLoader).copy()
    params['model_id'] = 0

    #Parameter manager
    pm = parameter_manager.Parameter_Manager(params=params)

    # for accessing the preprocessed data
    train_path = os.path.join(params['path_root'], params['path_data'], 'train')
    valid_path = os.path.join(params['path_root'], params['path_data'], 'valid')
    
    # Load data
    if pm.arch == 0: # MLP
        save_path = os.path.join(params['path_root'], params['path_data'], 'dataset.pt')
        load_pickle_data(train_path, valid_path, save_path, arch='mlp')
    elif pm.arch == 1 or params['arch'] == 2: # LSTM
        save_path = os.path.join(params['path_root'], params['path_data'], 'slices_dataset.pt')
        load_pickle_data(train_path, valid_path, save_path, arch='lstm')
    else:
        logging.error("datamodule.py | Dataset {} not implemented!".format(params['which']))
        exit()