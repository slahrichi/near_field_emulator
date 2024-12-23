import os
import argparse
import yaml

from core import train, preprocess_data, compile_data, modes
from conf.schema import load_config
from evaluation import eval_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help = "config.yaml specifiying experiment parameters")
    args = parser.parse_args()
    
    # Load parameters from the specified config YAML
    #params = yaml.load(open(args.config), Loader = yaml.FullLoader).copy()
    #directive = params['directive'] # WALL-E reference?
    
    # Load the MainConfig object
    config = load_config(args.config)
    directive = config.directive
    
    if directive == 0:
        print("Training model...\n")
        train.run(config)
    elif directive == 1:
        print('Evaluating model...\n')
        eval_model.run(config)
    elif directive == 2:
        raise NotImplementedError('Loading results not fully implemented yet.')
    elif directive == 3:
        raise NotImplementedError('MEEP simulation process not fully implemented yet.')
    elif directive == 4:
        print('Preprocessing DFT volumes...')
        preprocess_data.run(config)
        print('Compiling data into .pt file...')
        compile_data.run(config.paths)
    elif directive == 5:
        print(f"Encoding {config.model.modelstm['method']} modes...")
        modes.run(config)
    else:
        raise NotImplementedError(f'config.yaml: directive: {directive} is not a valid directive.')