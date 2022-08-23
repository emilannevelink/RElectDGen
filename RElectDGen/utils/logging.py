import os
import json
import pandas as pd
import numpy as np
from .save import get_results_dir

def write_to_tmp_dict(filename,dict):
    
    if os.path.isfile(filename):
        with open(filename,'r') as fl:
            old_dict = json.load(fl)
    else:
        old_dict = {}

    new_dict = {**old_dict,**dict}

    with open(filename,'w') as fl:
        json.dump(new_dict,fl)


def get_mae_from_results(root='results', index = None):

    results_dir = get_results_dir(root=root)

    results_metrics = os.path.join(results_dir,'metrics_epoch.csv')
    data = pd.read_csv(results_metrics)

    best_ind = np.argmin(data[' validation_loss'])

    if index is None:
        mae_dict = {
            'training_f_mae': float(data['training_loss_f'][best_ind]),
            'training_e_mae': float(data['training_e_mae'][best_ind]),
            'validation_f_mae': float(data['validation_loss_f'][best_ind]),
            'validation_e_mae': float(data['validation_e_mae'][best_ind]),
            'best_training_loss': float(np.min(data['training_loss'])),
            'best_validation_loss': float(np.min(data['validation_loss'])),
        }
    else:
        mae_dict = {
            f'training_f_mae_{index}': float(data['training_loss_f'][best_ind]),
            f'training_e_mae_{index}': float(data['training_e_mae'][best_ind]),
            f'validation_f_mae_{index}': float(data['validation_loss_f'][best_ind]),
            f'validation_e_mae_{index}': float(data['validation_e_mae'][best_ind]),
            f'best_training_loss_{index}': float(np.min(data['training_loss'])),
            f'best_validation_loss_{index}': float(np.min(data['validation_loss'])),
        }

    return mae_dict

def UQ_params_to_dict(params, prefix: str = ''):
    if not prefix.endswith('_'):
        prefix += '_'
    out_dict = {}
    for key in params:
        if isinstance(params[key],list):
            out_dict[prefix+key+'_base'] = float(params[key][0][0])
            out_dict[prefix+key+'_linear'] = float(params[key][0][1])
        else:
            out_dict[prefix+key+'_base'] = float(params[key][0])
            out_dict[prefix+key+'_linear'] = float(params[key][1])
    
    return out_dict

def add_checks_to_config(config, checks):
    if config.get('checks') is not None:
        config_checks = config['checks']
    else:
        config_checks = {}

    for key in checks:
        if key in config_checks:
            config_checks[key].append(checks[key])
        else:
            config_checks[key] = [checks[key]]

    config['checks'] = config_checks
    return config

def get_dataset_sizes(config, tmp_filename):
    if not os.path.isfile(tmp_filename):
        return 1, 0 #just return two numbers that aren't the same
    
    with open(tmp_filename,'r') as fl:
        tmp_dict = json.load(fl)

    current_dataset_size = tmp_dict.get('dataset_size',0)

    try:
        log_filename = os.path.join(config.get('data_directory'),config.get('log_filename'))
    except:
        return 1,0
    if os.path.isfile(log_filename):
        logcsv = pd.read_csv(log_filename)
        last_dataset_size = logcsv['dataset_size'].values[-1]
    else:
        last_dataset_size = current_dataset_size-1

    return last_dataset_size, current_dataset_size