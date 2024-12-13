import logging
import torch
import yaml
import traceback
import sys
import os

from utils.mapping import get_model_type

# debugging
#logging.basicConfig(level=logging.DEBUG)

class Parameter_Manager():
    def __init__(self, config = None,  params = None):
        logging.debug("parameter_manager.py - Initializing Parameter_Manager")

        if config is not None:
            self.open_config(config)
        if params is not None:
            self.params = params

        self.parse_params(self.params)

    def open_config(self, config_file):
        try:
            with open(config_file) as c:
                self.params = yaml.load(c, Loader = yaml.FullLoader)
        except Exception as e:
            logging.error(e)
            sys.exit()
            
    def parse_params(self, params):
        try:
            logging.debug(f"Keys in params: {params.keys()}")

            self.experiment = params['experiment']

            # Load: Paths 
            self.path_root = params['path_root']
            self.path_data = params['path_data']
            self.path_train = params['path_train']
            self.path_valid = params['path_valid']
            self.path_results = params['path_results']
            self.path_resims = params['path_resims']
            self._path_checkpoint = params['path_checkpoint']
            self.path_pretrained_ae = params['path_pretrained_ae']
 
            # Load: Trainer Params
            self.batch_size = params['batch_size']
            self.num_epochs = params['num_epochs']
            self.valid_rate = params['valid_rate']
            self.accelerator = params['accelerator']
            self.gpu_flag, self.gpu_list = params['gpu_config']
            self.patience = params['patience']
            self.min_delta = params['min_delta']
            self.include_testing = params['include_testing']
            self.cross_validation = params['cross_validation']
            
            # Load: Model Params
            self.weights = params['weights']
            self.optimizer = params['optimizer']
            self.lr_scheduler = params['lr_scheduler']
            self._mcl_params = params['mcl_params']
            self.num_classes = params['num_classes']
            self.num_design_params = params['num_design_params']
            self.learning_rate = params['learning_rate']
            self.transfer_learn = params['transfer_learn']
            self.load_checkpoint = params['load_checkpoint']
            self.objective_function = params['objective_function']
            self._arch = params['arch']
            self.mlp_real = params['mlp_real']
            self.mlp_imag = params['mlp_imag']
            self.mlp_strategy = params['mlp_strategy']
            self.patch_size = params['patch_size']
            self.lstm = params['lstm']
            self.convlstm = params['convlstm']
            self.seq_len = params['seq_len']
            self.io_mode = params['io_mode']
            self.autoreg = params['autoreg']
            self.spacing_mode = params['spacing_mode']
            self.cvnn = params['cvnn']
            self.autoencoder = params['autoencoder']
            self.modelstm = params['modelstm']
            # Load: Datamodule Params
            self.n_cpus = params['n_cpus']
            self.n_folds = params['n_folds']
            self.interpolate_fields = params['interpolate_fields']
            
            # Load: Kube Params
            self.kube = params['kube']
            
            # Load: Physical Params
            self._distance = params['distance']
            if(not(isinstance(self._distance, torch.Tensor))):
                self._distance = torch.tensor(self._distance)
            self._wavelength = torch.tensor(float(params['wavelength']))
            # Propagator
            self.Nxp = params['Nxp']
            self.Nyp = params['Nyp']
            self.Lxp = params['Lxp']
            self.Lyp = params['Lyp']
            self._adaptive = params['adaptive']

            # Load: Metasurface Simulation Params
            self.Nx_metaAtom = params['Nx_metaAtom'] 
            self.Ny_metaAtom = params['Ny_metaAtom'] 
        
            self.Lx_metaAtom = params['Lx_metaAtom']
            self.Ly_metaAtom = params['Ly_metaAtom']

            self.n_fusedSilica = params['n_fusedSilica']
            self.n_PDMS = params['n_PDMS']
            self.n_amorphousSilica = params['n_amorphousSilica']

            self.h_pillar = params['h_pillar']
            self.thickness_pml = params['thickness_pml']
            self.thickness_fusedSilica = params['thickness_fusedSilica']
            self.thickness_PDMS = params['thickness_PDMS']
            
            # Datashape from the sim information
            self.data_shape = [1,2,self.Nxp,self.Nyp]

            # Determine the type of experiment we are running
            self.model_id = params['model_id']


            try:
                self.jobid = os.environ['SLURM_JOB_ID']
            except:
                self.jobid = 0
                
            self.model_type = get_model_type(self._arch)

            if self.model_type == 'modelstm': # further categorize by mode encoding method
                self.path_results = f"{self.path_results}meep_meep/{self.model_type}/{self.modelstm['method']}/model_{self.model_id}/"
            else:
                self.path_results = f"{self.path_results}meep_meep/{self.model_type}/model_{self.model_id}/"
            self.results_path = self.path_results

            self.seed_flag, self.seed_value = params['seed']
        
            self.collect_params()

        except Exception as e:
            logging.error(e)
            traceback.print_exc()
            sys.exit()

    def collect_params(self):
        logging.debug("Parameter_Manager | collecting parameters")
        self._params_model = {
                                'weights'               : self.weights,
                                'optimizer'             : self.optimizer,
                                'lr_scheduler'          : self.lr_scheduler,
                                'data_shape'            : self.data_shape,
                                'num_epochs'            : self.num_epochs,
                                'num_classes'           : self.num_classes,
                                'learning_rate'         : self.learning_rate,
                                'transfer_learn'        : self.transfer_learn, 
                                'path_checkpoint'       : self.path_checkpoint,
                                'load_checkpoint'       : self.load_checkpoint,
                                'objective_function'    : self.objective_function,
                                'mcl_params'            : self._mcl_params,
                                'num_design_params'     : self.num_design_params,
                                'arch'                  : self._arch,
                                'mlp_real'              : self.mlp_real,
                                'mlp_imag'              : self.mlp_imag,
                                'mlp_strategy'          : self.mlp_strategy,
                                'patch_size'            : self.patch_size,
                                'lstm'                  : self.lstm,
                                'convlstm'              : self.convlstm,
                                'seq_len'               : self.seq_len,
                                'io_mode'               : self.io_mode,
                                'autoreg'               : self.autoreg,
                                'spacing_mode'          : self.spacing_mode,
                                'path_pretrained_ae'    : self.path_pretrained_ae,
                                'cvnn'                  : self.cvnn,
                                'autoencoder'           : self.autoencoder,
                                'modelstm'              : self.modelstm,
                                'batch_size'            : self.batch_size,
                                }

             
        self._params_propagator = {
                                'Nxp'           : self.Nxp, 
                                'Nyp'           : self.Nyp, 
                                'Lxp'           : self.Lxp, 
                                'Lyp'           : self.Lyp,
                                'distance'      : self._distance,
                                'adaptive'      : self.adaptive,
                                'batch_size'    : self.batch_size,
                                'wavelength'    : self._wavelength, 
                                }
                
        self._params_datamodule = {
                                'Nxp'           : self.Nxp, 
                                'Nyp'           : self.Nyp, 
                                'n_cpus'        : self.n_cpus,
                                'path_root'     : self.path_root, 
                                'path_data'     : self.path_data, 
                                'batch_size'    : self.batch_size,
                                'n_folds'       : self.n_folds,
                                'seed'          : self.seed_value,
                                'seq_len'       : self.seq_len,
                                'arch'          : self._arch,
                                'mlp_strategy'  : self.mlp_strategy,
                                'patch_size'    : self.patch_size,
                                'interpolate_fields' : self.interpolate_fields,
                                'io_mode'       : self.io_mode,
                                'spacing_mode'  : self.spacing_mode,
                                'experiment'    : self.experiment,
                                'modelstm'      : self.modelstm,
                                }
        
        self._params_kube = {
                                'kube'          : self.kube,
                                'experiment'    : self.experiment,
                                'arch'          : self._arch,
                                'model_id'      : self.model_id,
                                }

        self._params_trainer = {
                            'num_epochs'        : self.num_epochs, 
                            'valid_rate'        : self.valid_rate,
                            'accelerator'       : self.accelerator,
                            'patience'          : self.patience,
                            'min_delta'         : self.min_delta,
                            'experiment'        : self.experiment,
                            'include_testing'   : self.include_testing,
                            'cross_validation'  : self.cross_validation
                            }

        self._params_meep = {
                            'Nx_metaAtom'           : self.Nx_metaAtom,
                            'Ny_metaAtom'           : self.Ny_metaAtom,
                            'Lx_metaAtom'           : self.Lx_metaAtom,
                            'Ly_metaAtom'           : self.Ly_metaAtom,

                            'n_fusedSilica'         : self.n_fusedSilica,
                            'n_PDMS'                : self.n_PDMS,
                            'n_amorphousSilica'     : self.n_amorphousSilica,

                            'h_pillar'              : self.h_pillar,
                            'thickness_pml'         : self.thickness_pml,
                            'thickness_fusedSilica' : self.thickness_fusedSilica,
                            'thickness_PDMS'        : self.thickness_PDMS,

                            'data_shape'            : self.data_shape,
                            'model_id'              : self.model_id,
                            }

        self._all_paths = {
                        'path_root'                     : self.path_root, 
                        'path_data'                     : self.path_data, 
                        'path_train'                    : self.path_train, 
                        'path_valid'                    : self.path_valid,
                        'path_results'                  : self.path_results, 
                        'path_checkpoint'               : self._path_checkpoint,
                        'path_resims'                   : self.path_resims,
                        'path_pretrained_ae'            : self.path_pretrained_ae,
                        }

    @property 
    def params_model(self):         
        return self._params_model


    @property
    def params_propagator(self):
        return self._params_propagator                         


    @property
    def params_datamodule(self):
        return self._params_datamodule
    
    @property
    def params_kube(self):
        return self._params_kube

    @property 
    def params_trainer(self):
        return self._params_trainer

    @property
    def all_paths(self):
        return self._all_paths 

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        logging.debug("Parameter_Manager | setting distance to {}".format(value))
        self._distance = value
        self.collect_params()
        
    @property
    def wavelength(self):
        return self._wavelength
    
    @wavelength.setter
    def wavelength(self, value):
        logging.debug("Parameter_Manager | setting wavelength to {}".format(value))
        self._wavelength = value
        self.collect_params()
    
    @property
    def path_checkpoint(self):
        return self._path_checkpoint

    @path_checkpoint.setter
    def path_checkpoint(self, value):
        logging.debug("Parameter_Manager | setting path_checkpoint to {}".format(value))
        self._path_checkpoint = value
        self.collect_params()

    @property
    def arch(self):
        return self._arch

    @arch.setter
    def arch(self, value):
        logging.debug("Parameter_Manager | setting arch to {}".format(value))
        self._arch = value
        self.collect_params()

    @property
    def adaptive(self):
        return self._adaptive

    @adaptive.setter
    def adaptive(self, value):
        logging.debug("Parameter_Manager | setting adaptive to {}".format(value))
        self._adaptive = value
        self.collect_params()

    @property
    def mcl_params(self):
        return self._mcl_params

    @mcl_params.setter
    def mcl_params(self, value):
        logging.debug("Parameter_Manager | setting mcl_params to {}".format(value))
        self._mcl_params = value
        self.collect_params()

if __name__ == "__main__":
    import yaml
    params = yaml.load(open('../config.yaml'), Loader=yaml.FullLoader)
    pm = Parameter_Manager(params = params)