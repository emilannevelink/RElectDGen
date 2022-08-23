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
    parser.add_argument('--allow_calibrate', type=bool, default=False,
                        help='active_learning configuration file')
    parser.add_argument('--replot', type=bool, default=False,
                        help='active_learning configuration file')
    args = parser.parse_args(argsin)

    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    return config, args.allow_calibrate, args.replot

def main(args=None):
    print('Starting timer', flush=True)
    start = time.time()
    MLP_dict = {}
    

    config, allow_calibrate, replot = parse_command_line(args)

    root_dir = os.path.join(
        config.get('directory'),
        config.get('run_dir'),
        config.get('train_directory')
    )

    train_root = config['train_directory']
    if train_root[-1] == '/':
        train_root = train_root[:-1]

    uncertainty_function = config.get('uncertainty_function', 'Nequip_latent_distance')

    if uncertainty_function in ['Nequip_ensemble']:
        n_ensemble = config.get('n_uncertainty_ensembles',4)
        train_dirs = []
        for i in range(n_ensemble):
            root_dir = train_root + f'_{i}'
            train_dirs.append(os.listdir(root_dir))
        
        print(train_dirs)
        for i, td in enumerate(zip(*train_dirs)):
            model = []
            MLP_config = []
            plot = True
            print(td)
            for j in range(n_ensemble):
                print(td[j])
                root_dir = train_root + f'_{j}'
                train_directory = os.path.join(
                    root_dir,
                    td[j]
                )
                config_final = os.path.join(
                    train_directory,
                    'config_final.yaml'
                )
                if j ==0:
                    plot_filename = os.path.join(
                        train_directory,
                        config.get('UQ_plot_filename','UQ_fit.png')
                    )
                if not os.path.isfile(config_final) or (os.path.isfile(plot_filename) and not replot):
                    plot = False
                else:
                    calc_nn, mod, conf = nn_from_results(train_directory=train_directory)
                    model.append(mod)
                    MLP_config.append(conf)
            if plot:

                UQ_func = getattr(uncertainty_models,)

                UQ = UQ_func(model, config, MLP_config)
                UQ.calibrate()
                
                UQ.plot_fit(plot_filename)
    else:
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
            plot_filename = os.path.join(
                train_directory,
                config.get('UQ_plot_filename','UQ_fit.png')
            )
            if os.path.isfile(config_final) and (replot or not os.path.isfile(plot_filename)):
                if os.path.isdir(uncertainty_dir) or allow_calibrate:
                    try:
                        calc_nn, model, MLP_config = nn_from_results(train_directory)

                        UQ_func = getattr(uncertainty_models,)

                        UQ = UQ_func(model, config, MLP_config)
                        UQ.calibrate()
                        
                        UQ.plot_fit(plot_filename)
                    except Exception as e:
                        print(train_dir)
                        print(e)

if __name__ == "__main__":
    main()