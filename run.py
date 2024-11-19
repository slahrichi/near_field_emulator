import train
import os
import argparse
import yaml

def run(params):
    train.train(params)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help = "Experiment: config file")
    args = parser.parse_args()
    
    # Load parameters from the specified config YAML
    params = yaml.load(open(args.config), Loader = yaml.FullLoader).copy()
    
    # Run the experiment
    run(params)