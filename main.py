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
    
    if params['experiment'] == 0:
        print("Training model...")
        train.train(params)
    elif params['experiment'] == 1:
        print("Pretraining autoencoder...")
        train.train(params)
    elif params['experiment'] == 2:
        print('Compiling preprocessed pickle files...')
        compile_data(params)
    elif params['experiment'] == 3:
        raise NotImplementedError('Loading results not fully implemented yet.')
    elif params['experiment'] == 4:
        print('Evaluating model...')
        eval_model(params)
    else:
        raise NotImplementedError('Experiment not recognized.')