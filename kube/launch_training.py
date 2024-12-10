import os
import sys
import yaml
import time
import subprocess
import argparse
from IPython import embed

from dateutil.tz import tzutc
from kubernetes import client, config
from jinja2 import Environment, FileSystemLoader

sys.path.append("../")

from kube.support import exit_handler, load_file, save_file, parse_args, load_config
from utils import parameter_manager, mapping

def run(params):

    template = load_file(params['kube']['train_job']['paths']['template'])
    tag = params['kube']['train_job']['paths']['template'].split("/")[-1]
    folder = params['kube']['train_job']['paths']['template'].replace("/%s" % tag, "")
    environment = Environment(loader = FileSystemLoader(folder))
    template = environment.get_template(tag)
    
    model_type = mapping.get_model_type(params['arch'], params['experiment'])
        
    job_name = model_type + '-training'

    template_info = {'job_name': job_name,
                        'num_cpus': str(params['kube']['train_job']['num_cpus']),
                        'num_gpus': str(params['kube']['train_job']['num_gpus']),
                        'num_mem_req': str(params['kube']['train_job']['num_mem_req']),
                        'num_mem_lim': str(params['kube']['train_job']['num_mem_lim']),
                        'pvc_preprocessed': params['kube']['pvc_preprocessed'],
                        'pp_data_path': params['kube']['compile_job']['paths']['data']['preprocessed_data'],
                        'pvc_results': params['kube']['pvc_results'],
                        'results_path': params['kube']['train_job']['paths']['results']['model_results'],
                        'ckpt_path': params['kube']['train_job']['paths']['results']['model_checkpoints'],
                        'path_image': params['kube']['image'],
                    }

    filled_template = template.render(template_info)

    if not os.path.exists(params['kube']['job_files']):
        os.makedirs(params['kube']['job_files'])
    path_job = os.path.join(params['kube']['job_files'], job_name + ".yaml")
    save_file(path_job, filled_template)

    #subprocess.run(['kubectl', 'apply', '-f', path_job])
    #print(f"launching job for {arch}, {sequence}")
         
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help = "Experiment: config file")
    args = parser.parse_args()

    params = load_config(args.config)
    
    pm = parameter_manager.Parameter_Manager(params=params)

    run(pm.params_kube)