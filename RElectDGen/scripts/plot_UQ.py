import argparse
import os
import yaml
from ..calculate.calculator import nn_from_results
import time

from ..uncertainty import models as uncertainty_models

def parse_command_line(argsin):
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='config_file', type=str,
                        help='active_learning configuration file')
    parser.add_argument('allow_calibrate', type=bool, default=False,
                        help='active_learning configuration file')
    args = parser.parse_args(argsin)

    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    return config, args.allow_calibrate

def main(args=None):
    print('Starting timer', flush=True)
    start = time.time()
    MLP_dict = {}
    

    config, allow_calibrate = parse_command_line(args)

    root_dir = os.path.join(
        config.get('run_dir'),
        config.get('train_directory')
    )

    for train_dir in os.listdir(root_dir):
        train_directory = os.path.join(
            root_dir,
            train_dir
        )
        config_final = os.path.join(
            train_directory,
            'config_final.yaml'
        )
        uncertainty_dir = os.path.join(
            train_directory,
            'uncertainty'
        )

        if os.path.isfile(config_final):
            if os.path.isdir(uncertainty_dir) or allow_calibrate:
                calc_nn, model, MLP_config = nn_from_results(train_directory)

                UQ_func = getattr(uncertainty_models,config.get('uncertainty_function', 'Nequip_latent_distance'))

                UQ = UQ_func(model, config, MLP_config)
                UQ.calibrate()
                plot_filename = os.path.join(
                    train_directory,
                    config.get('UQ_plot_filename','UQ_fit.png')
                )
                UQ.plot_fit(plot_filename)

if __name__ == "__main__":
    main()