import yaml
import argparse
from typing import Dict, List, Tuple


def parse_datasetblob_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for creating an InMemoryDatasetBlob.

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


def collect_datasetblob_configs(
        args: argparse.Namespace
    ) -> Tuple[Dict, Dict]:
    """
    Collect configurations from the config file and command line arguments for creating an InMemoryDatasetBlob.

    Parameters:
    ----------
    - args (argparse.Namespace): The command line arguments.

    Returns:
    ----------
    - Tuple[Dict, Dict]: The config dictionary.
    """
    # --------------------- Process Main Config ---------------------
    # Get the config file name from command line argument
    config_fname = args.config_file

    # Read parameters from config file
    with open(config_fname, "r") as f:
        config = yaml.safe_load(f)

    # --------------------- Override Parameters ---------------------
    # Override parameters from command line
    if args.override:
        for arg in args.override:
            key, value = arg.split("=")
            if key == 'overwrite':
                if value == 'True':
                    config['dataset']['overwrite'] = True
                elif value == 'False':
                    config['dataset']['overwrite'] = False

    return config