import argparse

import yaml
from ..calculate.calculator import nn_from_results
import time

from ..uncertainty import models as uncertainty_models

def parse_command_line(argsin):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config',
                        help='active_learning configuration file', type=str)
    args = parser.parse_args(argsin)

    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    return config

def main(args=None):
    print('Starting timer', flush=True)
    start = time.time()
    MLP_dict = {}
    

    config = parse_command_line(args)
    
    calc_nn, model, MLP_config = nn_from_results()

    UQ_func = getattr(uncertainty_models,config.get('uncertainty_function', 'Nequip_latent_distance'))

    UQ = UQ_func(model, config, MLP_config)
    UQ.calibrate()
    plot_filename = config.get('UQ_plot_filename','plots/fit.png')
    UQ.plot_fit(plot_filename)

if __name__ == "__main__":
    main()