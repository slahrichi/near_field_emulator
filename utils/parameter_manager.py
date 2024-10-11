import logging
import torch
import yaml
import traceback
import sys
import os

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
            # Load: Paths 
            self.path_root = params['path_root']
            self.path_data = params['path_data']
            self.path_model = params['path_model']
            self.path_train = params['path_train']
            self.path_valid = params['path_valid']
            self.path_results = params['path_results']
            self.path_resims = params['path_resims']
            self._path_checkpoint = params['path_checkpoint']
 
            # Load: Trainer Params
            self.batch_size = params['batch_size']
            self.num_epochs = params['num_epochs']
            self.valid_rate = params['valid_rate']
            self.accelerator = params['accelerator']
            self.gpu_flag, self.gpu_list = params['gpu_config']

            # Load: Model Params
            self.weights = params['weights']
            self.backbone = params['backbone']
            self.optimizer = params['optimizer']
            self._mcl_params = params['mcl_params']
            self.num_classes = params['num_classes']
            self.num_design_params = params['num_design_params']
            self.learning_rate = params['learning_rate']
            self.transfer_learn = params['transfer_learn']
            self.load_checkpoint = params['load_checkpoint']
            self.objective_function = params['objective_function']
            
            # Load: MLP params
            self._mlp_real = params['mlp_real']
            self._mlp_imag = params['mlp_imag']

            # Load: Datamodule Params
            self._which = params['which']
            self.n_cpus = params['n_cpus']
            
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

            self.path_model = f"{self.path_model}/model_{self.model_id}/"
            self.path_results = f"{self.path_results}/model_{self.model_id}/"
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
                                'backbone'              : self.backbone,
                                'optimizer'             : self.optimizer,
                                'data_shape'            : self.data_shape,
                                'num_classes'           : self.num_classes,
                                'learning_rate'         : self.learning_rate,
                                'transfer_learn'        : self.transfer_learn, 
                                'path_checkpoint'       : self.path_checkpoint,
                                'load_checkpoint'       : self.load_checkpoint,
                                'objective_function'    : self.objective_function,
                                'mcl_params'            : self._mcl_params,
                                'num_design_params'     : self.num_design_params,
                                'mlp_real'              : self._mlp_real,
                                'mlp_imag'              : self._mlp_imag,
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
                                'which'         : self._which,
                                'n_cpus'        : self.n_cpus,
                                'path_root'     : self.path_root, 
                                'path_data'     : self.path_data, 
                                'batch_size'    : self.batch_size, 
                                }

        self._params_trainer = {
                            'num_epochs'        : self.num_epochs, 
                            'valid_rate'        : self.valid_rate,
                            'accelerator'       : self.accelerator, 
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
                        'path_model'                    : self.path_model,
                        'path_train'                    : self.path_train, 
                        'path_valid'                    : self.path_valid,
                        'path_results'                  : self.path_results, 
                        'path_model'                    : self.path_model, 
                        'path_results'                  : self.path_results, 
                        'path_checkpoint'               : self._path_checkpoint,
                        'path_resims'                   : self.path_resims
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
    def which(self):
        return self._which

    @which.setter
    def which(self, value):
        logging.debug("Parameter_Manager | setting which to {}".format(value))
        self._which = value
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
    print(pm.path_model)