import argparse


def main(args=None):
    config = parse_command_line(args)

    if config.get('machine','arjuna')=='arjuna':
        from ..utils.generate_shell_arjuna import shell_from_config
    elif config.get('machine')=='bridges':
        from ..utils.generate_shell_bridges import shell_from_config

    shell_from_config(config)

def parse_command_line(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config',
                        help='active_learning configuration file', type=str)
    args = parser.parse_args()
    import yaml

    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)

    return config

if __name__ == "__main__":
    main()