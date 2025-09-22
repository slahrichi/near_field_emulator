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
from tqdm import tqdm

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
sys.path.append('../')
from core import curvature
from utils import mapping

# debugging
#logging.basicConfig(level=logging.DEBUG)

class NF_Datamodule(LightningDataModule):
    def __init__(self, conf, transform = None):
        super().__init__() 
        logging.debug("datamodule.py - Initializing NF_DataModule")

        self.conf = conf.copy()
        logging.debug("datamodule.py - Setting conf to {}".format(self.conf))
        self.directive = conf.directive
        self.model_type = conf.model.arch
        self.n_cpus = conf.data.n_cpus
        self.seed = conf.seed
        self.n_folds = conf.data.n_folds
        self.mlp_strategy = conf.model.mlp_strategy
        self.patch_size = conf.model.patch_size
        self.arch = conf.model.arch

        self.path_data = conf.paths.data
        logging.debug("datamodule.py - Setting path_data to {}".format(self.path_data))
        self.batch_size = conf.trainer.batch_size
        self.transform = transform #TODO
        self.index_map = {
            'train': [],
            'valid': []
        }
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

    # TODO: Getting a bit messy, consider abstraction/subclassing
    def setup(self, stage: Optional[str] = None):
        if self.conf.data.source == 'projections':
            self.setup_projections()
        else:
            self.setup_fields()

    def setup_projections(self, stage=None):
        """Load projections from PVC and radii from the main dataset file."""
        print("Setting up projection-based dataset...")

        # 1. Load projections from the pickle file in the PVC
        projections_file = os.path.join(f"/data/{self.conf.kube.pvc_projections}/", "projections_0.pkl")
        try:
            with open(projections_file, 'rb') as f:
                projections_data = pickle.load(f)
            print(f"Successfully loaded projections from {projections_file}")
            
            # Correctly slice each projection array in the list
            num_projections_to_use = self.conf.data.num_projections
            sliced_projections = [p[:num_projections_to_use] for p in projections_data['projections']]
            
            # Convert the list of sliced arrays into a single 2D tensor
            projections = torch.tensor(np.array(sliced_projections), dtype=torch.float32)
            
        except Exception as e:
            print(f"ERROR: Failed to load or process projections file {projections_file}: {e}")
            return

        # 2. Load the original dataset to get the radii and train/valid tags
        datapath = self.get_datapath()
        try:
            data = torch.load(datapath)
            print(f"Successfully loaded original dataset from {datapath} to get radii.")
            radii = data['radii']
            tags = data['tag']
        except Exception as e:
            print(f"ERROR: Failed to load original dataset file {datapath}: {e}")
            return

        # 3. Ensure projections and radii have the same number of samples
        num_samples = min(len(projections), len(radii))
        if len(projections) != len(radii):
            print(f"Warning: Mismatch in number of samples. Projections: {len(projections)}, Radii: {len(radii)}. Using {num_samples} samples.")
        
        projections = projections[:num_samples]
        radii = radii[:num_samples]
        tags = tags[:num_samples]
        print(f"Using {num_samples} consistent samples for training.")

        # 4. Create the full dataset
        self.dataset = ProjectionsDataset(projections, radii)

        # 5. Create a map of indices for the original train/valid split
        for i in range(len(tags)):
            if tags[i] == 0:
                self.index_map['valid'].append(i)
            else:
                self.index_map['train'].append(i)
        print(f"Loaded {len(self.index_map['train'])} training samples and {len(self.index_map['valid'])} validation samples.")

    def setup_fields(self):
        datapath = self.get_datapath()
        data = torch.load(datapath)
        if self.conf.data.subset:
            data = self.subset_data(data)
        if self.model_type == 'autoencoder': # pretraining
            self.dataset = format_ae_data(data, self.conf)
        elif self.model_type in ['mlp', 'cvnn', 'inverse', 'NA', 'convTandem']:
            if self.conf.model.interpolate_fields: # interpolate fields to lower resolution
                data = interpolate_fields(data)
            self.dataset = WaveMLP_Dataset(data, self.transform, self.mlp_strategy, self.patch_size, buffer=self.conf.data.buffer, arch=self.arch)
        else: # time series models
            self.dataset = format_temporal_data(data, self.conf)
        # create a map of indices for OG train/valid split - default for when we don't use crossval
        for i in range(len(data['tag'])):
            if data['tag'][i] == 0:
                self.index_map['valid'].append(i)
            else:
                self.index_map['train'].append(i)
                
    def subset_data(self, data):
        N = self.conf.data.subset
        total_samples = len(data[next(iter(data))])
        assert N <= total_samples, f"Subset size {N} exceeds dataset size {total_samples}"
        subset_indices = torch.randperm(total_samples)[:N]
        subset_dict = {key: value[subset_indices] for key, value in data.items()}
        logging.debug("NF_DataModule | Using {} out of {} available training samples".format(N, total_samples))
        return subset_dict

    def get_datapath(self):
        """Based on params, return the correct dataset we'll be using"""
        if not self.conf.data.buffer:
            return os.path.join(self.path_data, 'preprocessed_data', 'dataset_nobuffer.pt')
        elif self.conf.model.arch == 'modelstm':
            return os.path.join(self.path_data, 'preprocessed_data', f"dataset_{self.conf.model.modelstm.method}.pt")
        else:
            #wv = str(self.conf.data.wavelength).replace('.', '')
            return os.path.join(self.path_data, 'preprocessed_data', f'dataset.pt')
        
    def setup_fold(self, train_idx, val_idx):
        # create subsets for the current fold
        self.train = Subset(self.dataset, train_idx)
        self.valid = Subset(self.dataset, val_idx)
        
    def setup_og(self):
        # use index map to create subsets in line with the original fixed random split
        self.train = Subset(self.dataset, self.index_map['train'])
        self.valid = Subset(self.dataset, self.index_map['valid'])

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

class ProjectionsDataset(Dataset):
    def __init__(self, projections, radii):
        self.projections = projections
        self.radii = radii

    def __len__(self):
        return len(self.projections)

    def __getitem__(self, idx):
        return self.projections[idx], self.radii[idx]

class WaveMLP_Dataset(Dataset):
    """
    Dataset for the MLP models associated with mapping design conf to fields.
    """
    def __init__(self, data, transform, approach=0, patch_size=1, buffer=True, arch=None):
        logging.debug("datamodule.py - Initializing WaveMLP_Dataset")
        self.transform = transform
        logging.debug("NF_Dataset | Setting transform to {}".format(self.transform))
        self.approach = approach
        self.patch_size = patch_size
        self.data = data
        self.is_buffer = buffer
        self.arch = arch
        # setup data accordingly
        self.format_data()
        
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
        near_field = self.near_fields[idx] # [2, 166, 166] or [2, 166, 166, 63] if self.approach == 3
        radius = self.radii[idx].float() # [9]
        
        if self.approach == 2:
            # selecting patch_size evenly distributed pixels
            x_indices, y_indices = zip(*self.distributed_indices)
            logging.debug(f"WaveMLP_Dataset | x_indices: {x_indices}, y_indices: {y_indices}")
            near_field = near_field[:, x_indices, y_indices]
            near_field = near_field.reshape(2, self.patch_size, self.patch_size)
        if self.transform:   
            near_field = self.transform(near_field)        
        logging.debug(f"WaveMLP_Dataset | near_field shape: {near_field.shape}")
        if self.arch in ["inverse", 'NA', 'convTandem']:
            logging.debug(f"WaveMLP_Dataset | Inverse arch; returning near_field as input; shape: {near_field.shape}")
            return near_field, radius

        return radius, near_field
    
    def format_data(self):
        self.radii = self.data['radii']
        self.phases = self.data['phases']
        self.derivatives = self.data['derivatives']
        if not self.is_buffer: # old buffer dataset (U-NET data)
            # focus on 1550 wavelength in y for now
            temp_nf_1550 = self.data['all_near_fields']['near_fields_1550']
            temp_nf_1550 = torch.stack(temp_nf_1550, dim=0) # stack all sample tensors
            temp_nf_1550 = temp_nf_1550.squeeze(1) # remove redundant dimension
            temp_nf_1550 = temp_nf_1550[:, 1, :, :, :] # [num_samples, mag/phase, 166, 166]
            # convert to cartesian coords
            mag, phase = mapping.polar_to_cartesian(temp_nf_1550[:, 0, :, :], temp_nf_1550[:, 1, :, :])
            mag = mag.unsqueeze(1)
            phase = phase.unsqueeze(1)
            self.near_fields = torch.cat((mag, phase), dim=1) # [num_samples, r/i, 166, 166]
        else: # using newer dataset (dataset.pt), simulated for time series models
            temp_nf_1550 = self.data['near_fields'] # [num_samples, 2, 166, 166, 63]
            if self.approach == 3:
                # loading all slices
                self.near_fields = temp_nf_1550 # [num_samples, 2, 166, 166, 63]
                logging.debug(f"WaveMLP_Dataset | format_data: grabbing all slices; near_fields shape: {self.near_fields.shape}")
            else:
                # grab the final slice
                self.near_fields = temp_nf_1550[:, :, :, :, -1] # [num_samples, 2, 166, 166]
                logging.debug(f"WaveMLP_Dataset | format_data: grabbing last slice; near_fields shape: {self.near_fields.shape}")
                

class WaveModel_Dataset(Dataset):
    """
    Dataset for the time series models associated with emulating wave propagation.
    """
    def __init__(self, samples, labels):
        logging.debug("datamodule.py - Initializing WaveModel_Dataset")
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return len(self.samples)
        
def select_data(conf):
    return NF_Datamodule(conf)

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
        # create path if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
        # keeping track of original split
        normalized_path = os.path.normpath(path)
        parent_dir = os.path.basename(normalized_path)
        is_train = parent_dir == 'train'
        current_tag = 1 if is_train else 0
        
        for current_file in tqdm(os.listdir(path),
                                desc="Compiling data - {}".format(path),
                                ncols=80,
                                file=sys.stdout,
                                mininterval=1.0): # loop through pickle files
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
    
def format_temporal_data(data, conf, order=(-1, 0, 1, 2)):
    """Formats the preprocessed data file into the correct setup  
    and order for the LSTM model.

    Args:
        data (tensor): the dataset
        conf (dict): configuration parameters
        order (tuple, optional): order of the sequence to be used. Defaults to (-1, 0, 1, 2).
        
    Returns:
        dataset (WaveModel_Dataset): formatted dataset
    """
    all_samples, all_labels = [], []
    spacing_mode = conf.model.spacing_mode
    io_mode = conf.model.io_mode
    seq_len = conf.model.seq_len
    # [samples, 2, xdim, ydim, 63] --> access each of the datapoints
    for i in range(data['near_fields'].shape[0]):
        full_sequence = data['near_fields'][i] # [2, xdim, ydim, total_slices]

        total = full_sequence.shape[-1] # all time slices
        
        if spacing_mode == 'distributed':
            if io_mode == 'one_to_many':
                # calculate seq_len+1 evenly spaced indices
                indices = np.linspace(1, total-1, seq_len+1)
                distributed_block = full_sequence[:, :, :, indices]
                # the sample is the first one, labels are the rest
                sample = distributed_block[:, :, :, :1]  # [2, xdim, ydim, 1]
                label = distributed_block[:, :, :, 1:]  # [2, xdim, ydim, seq_len]
                
            elif io_mode == 'many_to_many':
                # Calculate seq_len+1 evenly spaced indices for input and shifted output
                indices = np.linspace(0, total-1, seq_len+1).astype(int)
                distributed_block = full_sequence[:, :, :, indices]
                
                # Input sequence: all but last timestep
                sample = distributed_block[:, :, :, :-1]  # [2, xdim, ydim, seq_len]
                # Output sequence: all but first timestep
                label = distributed_block[:, :, :, 1:]   # [2, xdim, ydim, seq_len]
                
            else:
                # many to one, one to one not implemented
                raise NotImplementedError(f'Specified recurrent input-output mode is not implemented.')
            
            # rearrange dims and add to lists
            sample = sample.permute(order) # [1, 2, xdim, ydim]
            label = label.permute(order) # [seq_len, 2, xdim, ydim]
            all_samples.append(sample)
            all_labels.append(label)
            
        elif spacing_mode == 'sequential':
            if io_mode == 'one_to_many':
                #for t in range(0, total, conf['seq_len']+1): note: this raise the total number of sample/label pairs
                t = 0
                # check if there are enough timesteps for a full block
                if t + seq_len < total:
                    block = full_sequence[:, :, :, t:t+seq_len + 1]
                    # ex: sample -> t=0 , label -> t=1, t=2, t=3 (if seq_len were 3)
                    sample = block[:, :, :, :1]
                    label = block[:, :, :, 1:]
                        
            elif io_mode == 'many_to_many':
                # true many to many
                sample = full_sequence[:, :, :, :seq_len]
                label = full_sequence[:, :, :, 1:seq_len+1]
                
                # this is our 'encoder-decoder' mode - not really realistic here
                '''step_size = 2 * conf['seq_len']
                #for t in range(0, total, step_size):
                t = 0
                # check if there's enough
                if t + step_size <= total:
                    # input is first seq_len steps in the block
                    sample = full_sequence[:, :, :, t:t+conf['seq_len']]
                    # output is next seq_len steps
                    label = full_sequence[:, :, :, t+conf['seq_len']:t+step_size]'''
                
            else:
                raise NotImplementedError(f'Specified recurrent input-output mode is not implemented.')
                
            sample = sample.permute(order)
            label = label.permute(order)
            all_samples.append(sample)
            all_labels.append(label)
        
        else:
            # no other spacing modes are implemented
            raise NotImplementedError(f'Specified recurrent dataloading confuration is not implemented.')
        
    return WaveModel_Dataset(all_samples, all_labels)

def format_ae_data(data, conf):
    """Formats the preprocessed data file into the correct setup  
    and order for the autoencoder pretraining.

    Args:
        data (tensor): the dataset
        conf (dict): configuration parameters
        
    Returns:
        dataset (WaveModel_Dataset): formatted dataset
    """
    all_samples = []
    
    # 100 samples, 63 slices per sample, 63*100 = 6300 samples/labels to train on
    for i in range(data['near_fields'].shape[0]):
        full_sequence = data['near_fields'][i] # [2, xdim, ydim, total_slices]
        for t in range(full_sequence.shape[-1]):
            sample = full_sequence[:, :, :, t] # [2, xdim, ydim] single sample
            all_samples.append(sample)
            
    # were training on reconstruction, so samples == labels
    return WaveModel_Dataset(all_samples, all_samples)
        
def interpolate_fields(data):
    """Interpolates the fields to a lower resolution. Currently supports 2x downsampling.  

    Args:
        data (dict): dictionary containing the near fields, phases, and radii
        
    Returns:
        dataset (WaveModel_Dataset): formatted dataset
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