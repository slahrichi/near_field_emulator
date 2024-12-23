import os
import sys
import subprocess
import argparse
from jinja2 import Environment, FileSystemLoader

sys.path.append("../")
from kube.support import load_file, save_file
from conf.schema import load_config

def run(conf):

    template = load_file(conf.kube.evaluation_job['paths']['template'])
    tag = conf.kube.evaluation_job['paths']['template'].split("/")[-1]
    folder = conf.kube.evaluation_job['paths']['template'].replace("/%s" % tag, "")
    environment = Environment(loader = FileSystemLoader(folder))
    template = environment.get_template(tag)

    model_type = conf.model.arch
        
    job_name = f'ethan-{model_type}-eval'

    template_info = {'job_name': job_name,
                        'num_cpus': str(conf.kube.train_job['num_cpus']),
                        'num_gpus': str(conf.kube.train_job['num_gpus']),
                        'num_mem_req': str(conf.kube.train_job['num_mem_req']),
                        'num_mem_lim': str(conf.kube.train_job['num_mem_lim']),
                        'pvc_preprocessed': conf.kube['pvc_preprocessed'],
                        'pp_data_path': conf.kube.data_job['paths']['data']['preprocessed_data'],
                        'pvc_results': conf.kube['pvc_results'],
                        'results_path': conf.kube.train_job['paths']['results']['model_results'],
                        'ckpt_path': conf.kube.train_job['paths']['results']['model_checkpoints'],
                        'path_image': conf.kube['image'],
                    }

    filled_template = template.render(template_info)

    if not os.path.exists(conf.kube['job_files']):
        os.makedirs(conf.kube['job_files'])
    path_job = os.path.join(conf.kube['job_files'], job_name + ".yaml")
    save_file(path_job, filled_template)

    subprocess.run(['kubectl', 'apply', '-f', path_job])
         
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help = "Experiment: config file")
    args = parser.parse_args()

    conf = load_config(args.config)
    
    run(conf)