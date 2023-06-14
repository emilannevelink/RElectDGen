import argparse
from RElectDGen.shell.generate_shell import generate_shell

def main(args=None):
    config = parse_command_line(args)

    generate_shell(config)

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