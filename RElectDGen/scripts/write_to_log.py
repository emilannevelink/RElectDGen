import argparse, subprocess
from ase.parallel import world
import operator
import numpy as np
import yaml
import os
import json
import pandas as pd

def parse_command_line(argsin):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config',
                        help='active_learning configuration file', type=str)
    args = parser.parse_args(argsin)

    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    return config


def main(args = None):

    config = parse_command_line(args)

    #load tmp
    tmp_filename = os.path.join(config.get('directory'),config.get('run_dir'),config.get('tmp_file','tmp.json'))
    with open(tmp_filename,'r') as fl:
        tmp_dict = json.load(fl)

    new_line = pd.DataFrame(tmp_dict,index=[0])
    
    #create dataframe
    log_filename = os.path.join(config.get('data_directory'),config.get('log_filename'))
    if os.path.isfile(log_filename):
        dataframe = pd.read_csv(log_filename)

        #write line to dataframe
        dataframe = dataframe.append(new_line,ignore_index=True) 
    else:
        dataframe = new_line

    #save new dataframe
    dataframe.to_csv(log_filename,index=False)

    os.remove(tmp_filename)

if __name__ == "__main__":
    main()
                

