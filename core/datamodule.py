#--------------------------------
# Import: Basic Python Libraries
#--------------------------------
import os
import sys
import glob #TODO: Needed?
import torch
import logging
import numpy as np #TODO: Needed?
from typing import Optional
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
import pickle

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
sys.path.append('../')
#from core import custom_transforms as ct
#from core import preprocess_data
from core import curvature

class NF_Datamodule(LightningDataModule):
    def __init__(self, params, transform = None):
        super().__init__() 
        logging.debug("datamodule.py - Initializing CAI_DataModule")

        self.params = params.copy()
        self.n_cpus = self.params['n_cpus']

        self.path_data = self.params['path_data']
        self.path_root = self.params['path_root']
        self.path_data = os.path.join(self.path_root,self.path_data)
        logging.debug("datamodule.py - Setting path_data to {}".format(self.path_data))
        self.batch_size = self.params['batch_size']
       
        self.transform = transform #TODO

        self.initialize_cpus(self.n_cpus)

    def initialize_cpus(self, n_cpus):
        # Make sure default number of cpus is not more than the system has
        if n_cpus > os.cpu_count():
            n_cpus = 1
        self.n_cpus = n_cpus 
        logging.debug("NF_DataModule | Setting CPUS to {}".format(self.n_cpus))

    def prepare_data(self):
        pass
        #preprocess_data.preprocess_data(path = os.path.join(self.path_data, 'raw'))

    def setup(self, stage: Optional[str] = None):
        #TODO
        #train_file = 'preprocessed/cai_data_newInterpolate.pt'
        #valid_file = None
        #test_file = None
        data = torch.load(os.path.join(self.path_data, 'dataset.pt'))

        if stage == "fit" or stage is None:
            dataset = NF_Dataset(data, self.transform)
            train_set_size = int(len(dataset)*0.8)
            valid_set_size = len(dataset) - train_set_size
            self.train, self.valid = random_split(dataset, [train_set_size, valid_set_size])
        if stage == "test" or stage is None:
            #self.test = NF_Dataset(torch.load(os.path.join(self.path_data, test_file)), self.transform)
            self.test = self.valid

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.n_cpus, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, num_workers=self.n_cpus, shuffle=False, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.n_cpus, shuffle=False)

class NF_Dataset(Dataset):
    def __init__(self, data, transform):
        logging.debug("datamodule.py - Initializing customDataset")
        self.transform = transform
        logging.debug("customDataset | Setting transform to {}".format(self.transform))
        self.radii = data['radii']
        self.phases = data['phases']
        self.derivatives = data['derivatives']
        # various wavelengths, right now we'll just focus on 1550 in y
        self.all_near_fields = data['all_near_fields']
        
        # perform reshaping to utimately drop x and z dims
        temp_nf_1550 = self.all_near_fields['near_fields_1550']
        temp_nf_1550 = torch.stack(temp_nf_1550, dim=0) # stack all sample tensors
        temp_nf_1550 = temp_nf_1550.squeeze(1) # remove redundant dimension
        self.near_fields = temp_nf_1550[:, 1, :, :, :] # [num_samples, r/i, 166, 166]
        
        #self.transform = ct.per_sample_normalize()
        self.transform = None

    def __len__(self):
        return len(self.near_fields)

    def __getitem__(self, idx):
        if self.transform:   
            return self.transform(self.near_fields[idx]), self.radii[idx].float()
        else:   
            return self.near_fields[idx], self.radii[idx].float()
        
def select_data(params):
    if params['which'] == 'NFRP':
        return NF_Datamodule(params) 
    else:
        logging.error("datamodule.py | Dataset {} not implemented!".format(params['which']))
        exit()

#--------------------------------
# Initialize: Format data
#--------------------------------

def load_pickle_data(train_path, valid_path, save_path):
    near_fields = []
    phases = []
    derivatives = []
    
    for path in [train_path, valid_path]:
            for current_file in os.listdir(path): # loop through pickle files
                if current_file.endswith(".pkl"):
                    current_file_path = os.path.join(path, current_file)
                    
                    with open(current_file_path, "rb") as f:
                        data = pickle.load(f)
                        
                        # extracting final slice from meta_atom_rnn data (1550 wl)
                        near_field_slice = data['data'][:, :, :, -1].float()  # [2, 166, 166]
                        
                        # append near field and phase data
                        near_fields.append(near_field_slice)
                        phases.append(data['LPA phases'])
                        
                        # per phases, calculate derivatives and append
                        der = curvature.get_der_train(data['LPA phases'].view(1, 3, 3))
                        derivatives.append(der)
    
    # convert to tensors
    near_fields_tensor = torch.stack([torch.tensor(f) for f in near_fields], dim=0)  # [num_samples, 2, 166, 166]
    phases_tensor = torch.stack([torch.tensor(p) for p in phases], dim=0)  # [num_samples, 9]
    derivatives_tensor = torch.stack([torch.tensor(d) for d in derivatives], dim=0)  # [num_samples, 3, 3]    
    
    torch.save({'near_fields': near_fields_tensor, 'phases': phases_tensor, 'derivatives': derivatives_tensor}, save_path)
    print(f"Data saved to {save_path}")
    
#--------------------------------
# Initialize: Testing
#--------------------------------

if __name__=="__main__":
    import yaml
    import torch
    import matplotlib.pyplot as plt
    from utils import parameter_manager
    from pytorch_lightning import seed_everything

    logging.basicConfig(level=logging.DEBUG)
    seed_everything(1337)
    os.environ['SLURM_JOB_ID'] = '0'
    #plt.style.use(['science'])

    #Load config file   
    params = yaml.load(open('config.yaml'), Loader = yaml.FullLoader).copy()
    params['model_id'] = 0

    #Parameter manager
    pm = parameter_manager.Parameter_Manager(params=params)

    # for accessing the meta_atom_rnn data
    train_path = os.path.join(params['path_root'], params['path_data'], 'train')
    valid_path = os.path.join(params['path_root'], params['path_data'], 'valid')
    save_path = os.path.join(params['path_root'], params['path_data'], 'cai_data_file.pt')
    
    # Load data
    load_pickle_data(train_path, valid_path, save_path)

    #Initialize the data module
    dm = select_data(pm.params_datamodule)
    dm.prepare_data()
    dm.setup(stage="fit")