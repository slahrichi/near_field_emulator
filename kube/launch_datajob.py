#  Creating kubernetes job for compiling preprocessed data
# into a single PyTorch file

import os
import sys
import argparse
import yaml
import time
import subprocess

from dateutil.tz import tzutc
from kubernetes import client, config
from jinja2 import Environment, FileSystemLoader

sys.path.append("../")

from kube.support import exit_handler, load_file, save_file, parse_args, load_config
from utils import parameter_manager

def run(params):

    template = load_file(params['kube']['data_job']['paths']['template'])
    
    tag = params['kube']['data_job']['paths']['template'].split("/")[-1]
    folder = params['kube']['data_job']['paths']['template'].replace("/%s" % tag, "")
    environment = Environment(loader = FileSystemLoader(folder))
    template = environment.get_template(tag)

    job_name = f"ethan-{params['kube']['data_job']['kill_tag']}"

    template_info = {'job_name' : job_name,
                     'num_cpus' : str(params['kube']['data_job']['num_cpus']),
                     'num_mem_lim' : str(params['kube']['data_job']['num_mem_lim']),
                     'num_mem_req' : str(params['kube']['data_job']['num_mem_req']),
                     'pvc_preprocessed' : params['kube']['pvc_preprocessed'],
                     'preprocessed_path' : params['kube']['data_job']['paths']['data']['preprocessed_data'],
                     'path_image' : params['kube']['image']
                    }

    filled_template = template.render(template_info)

    if not os.path.exists(params['kube']['job_files']):
        os.makedirs(params['kube']['job_files'])
    path_job = os.path.join(params['kube']['job_files'], job_name + ".yaml")
    save_file(path_job, filled_template)

    subprocess.run(['kubectl', 'apply', '-f', path_job])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help = "Experiment: config file")
    args = parser.parse_args()

    params = load_config(args.config)
    
    pm = parameter_manager.Parameter_Manager(params=params)

    run(pm.params_kube)