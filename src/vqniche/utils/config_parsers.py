import yaml
import argparse
from typing import Dict


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
    ----------
    - argparse.Namespace: The command line arguments.
    """
    parser = argparse.ArgumentParser(description='Process command line arguments.')

    parser.add_argument('--config_file',
                        type=str,
                        help='Path to the config file')
    parser.add_argument('--override',
                        nargs='+',
                        help='Override parameters in the config file')

    return parser.parse_args()


def collect_configs(args: argparse.Namespace) -> Dict:
    """
    Collect configurations from the config file and command line arguments.

    Parameters:
    ----------
    - args (argparse.Namespace): The command line arguments.

    Returns:
    ----------
    - Dict: The configuration dictionary.
    """
    # Get the config file name from command line argument
    config_fname = args.config_file

    # Read parameters from config file
    with open(config_fname, "r") as f:
        config = yaml.safe_load(f)

    # Override parameters from command line
    if args.override:
        for arg in args.override:
            key, value = arg.split("=")
            if key == 'delaunay':
                if value == 'True':
                    config['data']['graph_kwargs'][key] = True
                elif value == 'False':
                    config['data']['graph_kwargs'][key] = False
            elif key == 'overwrite':
                if value == 'True':
                    config['experiment'][key] = True
                elif value == 'False':
                    config['experiment'][key] = False

    return config
