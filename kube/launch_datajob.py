#  Creating kubernetes job for compiling preprocessed data
# into a single PyTorch file

import os
import sys
import argparse
import subprocess
from jinja2 import Environment, FileSystemLoader

sys.path.append("../")
from kube.support import load_file, save_file
from conf.schema import load_config
def run(conf):

    template = load_file(conf.kube.data_job['paths']['template'])
    
    tag = conf.kube.data_job['paths']['template'].split("/")[-1]
    folder = conf.kube.data_job['paths']['template'].replace("/%s" % tag, "")
    environment = Environment(loader = FileSystemLoader(folder))
    template = environment.get_template(tag)

    job_name = f"ethan-{conf.kube.data_job['kill_tag']}"

    template_info = {'job_name' : job_name,
                     'num_cpus' : str(conf.kube.data_job['num_cpus']),
                     'num_mem_lim' : str(conf.kube.data_job['num_mem_lim']),
                     'num_mem_req' : str(conf.kube.data_job['num_mem_req']),
                     'pvc_preprocessed' : conf.kube.pvc_preprocessed,
                     'preprocessed_path' : conf.kube.data_job['paths']['data']['preprocessed_data'],
                     'path_image' : conf.kube.image
                    }

    filled_template = template.render(template_info)

    if not os.path.exists(conf.kube.job_files):
        os.makedirs(conf.kube.job_files)
    path_job = os.path.join(conf.kube.job_files, job_name + ".yaml")
    save_file(path_job, filled_template)

    #subprocess.run(['kubectl', 'apply', '-f', path_job])

if __name__=="__main__":
     
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help = "Experiment: config file")
    args = parser.parse_args()

    conf = load_config(args.config)
    
    run(conf)