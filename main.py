import os
import argparse
import yaml

from core import train, preprocess_data, compile_data, modes
from evaluation import eval_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help = "config.yaml specifiying experiment parameters")
    args = parser.parse_args()
    
    # Load parameters from the specified config YAML
    params = yaml.load(open(args.config), Loader = yaml.FullLoader).copy()
    directive = params['directive'] # WALL-E reference?
    
    if directive == 0:
        print("Training model...\n")
        train.run(params)
    elif directive == 1:
        print('Evaluating model...\n')
        eval_model.run(params)
    elif directive == 2:
        raise NotImplementedError('Loading results not fully implemented yet.')
    elif directive == 3:
        raise NotImplementedError('MEEP simulation process not fully implemented yet.')
    elif directive == 4:
        print('Preprocessing DFT volumes...')
        preprocess_data.run(params)
        print('Compiling data into .pt file...')
        compile_data.run(params)
    elif directive == 5:
        print(f"Encoding {params['modelstm']['method']} modes...")
        modes.run(params)
    else:
        raise NotImplementedError(f'config.yaml: directive: {directive} is not a valid directive.')