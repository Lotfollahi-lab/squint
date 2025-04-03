import copy
import yaml
import argparse
from typing import Dict, List, Tuple


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
    ----------
    - argparse.Namespace: The command line arguments.
    """
    parser = argparse.ArgumentParser(description='Process command line arguments.')

    parser.add_argument('--base_config_file',
                        type=str,
                        help='Path to the base config file')
    parser.add_argument('--sweep_config_files',
                        nargs='*',
                        help='Optional list of additional sweep config files')
    parser.add_argument('--override',
                        nargs='+',
                        help='Override parameters in the config file')

    return parser.parse_args()


def collect_configs(
        args: argparse.Namespace
    ) -> Tuple[Dict, Dict]:
    """
    Collect configurations from the config file and command line arguments.

    Parameters:
    ----------
    - args (argparse.Namespace): The command line arguments.

    Returns:
    ----------
    - Tuple[Dict, Dict]: The base config and sweep config dictionaries.
    """
    # --------------------- Process Main Config ---------------------
    # Get the config file name from command line argument
    base_config_fname = args.base_config_file

    # Read parameters from config file
    with open(base_config_fname, "r") as f:
        base_config = yaml.safe_load(f)

    # --------------------- Process Sweep Configs ---------------------
    # If sweep configs are provided, update the config with the sweep configs
    if args.sweep_config_files:
        base_config['experiment']['mode'] = 'sweep'
        sweep_config = prepare_sweep_config(args.sweep_config_files)
    else:
        base_config['experiment']['mode'] = 'standalone'
        sweep_config = {}

    # --------------------- Override Parameters ---------------------
    # Override parameters from command line
    if args.override:
        for arg in args.override:
            key, value = arg.split("=")
            if key == 'offline':
                if value == 'True':
                    base_config['logging']['offline'] = True
                elif value == 'False':
                    base_config['logging']['offline'] = False
            elif key == 'sweep_id':
                base_config['experiment'][key] = value
            elif key == 'run_cap':
                sweep_config['run_cap'] = int(value)

    return base_config, sweep_config


def prepare_sweep_config(
        sweep_config_files: List[str]
    ) -> Dict:
    """
    Prepare the sweep config from the sweep config files.

    Parameters:
    ----------
    - sweep_config_files (List[str]): The list of sweep config files.

    Returns:
    ----------
    - Dict: The sweep config dictionary.
    """
    sweep_config = {
        'method': None,
        'metric': None,
        'parameters': {},
    }

    for i, sweep_config_file in enumerate(sweep_config_files):
        with open(sweep_config_file, "r") as f:
            config = yaml.safe_load(f)

        if i == 0:
            sweep_config['method'] = config['method']
            sweep_config['metric'] = config['metric']
            sweep_config['name'] = config['name']
            sweep_config['run_cap'] = config['run_cap']
        else:
            sweep_config['name'] = f"{sweep_config['name']}_{config['name']}"

        sweep_config['parameters'].update(config['parameters'])

    return sweep_config


def update_config(
        base_config: Dict,
        current_run_params: Dict
    ) -> Dict:
    """
    Update the base config with the current run parameters.

    Parameters:
    ----------
    - base_config (Dict): The base config dictionary.
    - current_run_params (Dict): The current run parameters.

    Returns:
    ----------
    - Dict: The updated base config dictionary.
    """
    config = copy.deepcopy(base_config)
    for key, value in current_run_params.items():
        keys = key.split('.')
        current_level = config
        for k in keys[:-1]:
            current_level = current_level.setdefault(k, {})
        current_level[keys[-1]] = value

    return config