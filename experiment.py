import train
import os
import argparse
import yaml

def run(params):
    train.run(params)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help = "Experiment: config file")
    args = parser.parse_args()
    
    params = yaml.load(open(args.config), Loader = yaml.FullLoader).copy()
    
    run(params)