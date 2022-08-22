from ase.parallel import world
from nequip.utils import Config
import yaml, os
from ase.io import Trajectory

def update_config_trainval(config, fl_MLP_config):
    if world.rank == 0:

        MLP_config = Config.from_file(fl_MLP_config)

        combined_trajectory = os.path.join(config.get('data_directory'),config.get('combined_trajectory'))
        if os.path.isfile(combined_trajectory):
            traj = Trajectory(combined_trajectory)

        n_train = int(len(traj)*config['train_perc'])
        n_val = int(len(traj)*config['val_perc'])
        if n_val<config['n_val_min']:
            n_train -= (config['n_val_min']-n_val)
            n_val = config['n_val_min']
        MLP_config.update(dict(n_train=n_train))
        MLP_config.update(dict(n_val=n_val))
        with open(fl_MLP_config, "w+") as fp:
            yaml.dump(dict(MLP_config), fp)

def check_NN_parameters(MLP_config_new, MLP_config_old):
    parameters_to_check = [
        'model_builders',
        'num_layers',
        'chemical_embedding_irreps_out',
        'feature_irreps_hidden',
        'irreps_edge_sh',
        'conv_to_output_hidden_irreps_out',
        'nonlinearity_type',
        'nonlinearity_scalars',
        'nonlinearity_gates',
        'resnet',
        'num_basis',
        'invariant_layers',
        'invariant_neurons',
    ]

    value = True
    for parameter in parameters_to_check:
        if MLP_config_new[parameter] != MLP_config_old[parameter]:
            value = False
            print(f'{parameter} does not match in config files',flush=True)

    if value:
        MLP_config_new['workdir'] = MLP_config_old['workdir']

    # with open(fl_MLP_config_new, "w+") as fp:
    #     yaml.dump(dict(MLP_config_new), fp)

    return value

def get_results_dir(root='results'):
    max_time = 0
    for tmp in os.listdir(root):
        if not tmp.startswith('processed'):
            time_tmp = os.stat(os.path.join(root,tmp)).st_mtime
            if time_tmp>max_time:
                max_time = time_tmp
                train_directory = os.path.join(root,tmp)
    
    return train_directory