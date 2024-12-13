import os
import argparse
import yaml

import train
from utils.compile_data import compile_data
from evaluation.eval_model import eval_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help = "Experiment: config file")
    args = parser.parse_args()
    
    # Load parameters from the specified config YAML
    params = yaml.load(open(args.config), Loader = yaml.FullLoader).copy()
    directive = params['experiment'] # WALL-E reference?
    
    if directive == 0:
        print("Training model...\n")
        train.train(params)
    elif directive == 1:
        print('Evaluating model...\n')
        eval_model(params)
    elif directive == 2:
        raise NotImplementedError('Loading results not fully implemented yet.')
    elif directive == 3:
        raise NotImplementedError('MEEP simulation process not fully implemented yet.')
    elif directive == 4:
        print('Compiling preprocessed pickle files...')
        compile_data(params)
    else:
        raise NotImplementedError(f'config.yaml: experiment: {directive} is not a valid directive.')