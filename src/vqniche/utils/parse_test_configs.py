from typing import Dict, Optional, Literal

import os
import yaml
import argparse
from pathlib import Path

import numpy as np


def parse_test_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for testing.

    Returns:
    ----------
    - argparse.Namespace: The command line arguments.
    """
    parser = argparse.ArgumentParser(description='Process command line arguments.')

    parser.add_argument('--wandb_run_dir',
                        type=str,
                        help='Path to the wandb run directory')
    parser.add_argument('--model_ckpt_fname',
                        type=str,
                        default=None,
                        help='Optional model checkpoint file name')
    parser.add_argument('--metric_name',
                        type=str,
                        default='pearson_1hop_nbr',
                        help='Metric name to find the best checkpoint')
    parser.add_argument('--mode',
                        type=str,
                        default='max',
                        help='Mode to find the best checkpoint')
    parser.add_argument('--compute_metrics',
                        action='store_true',
                        help='Compute metrics')
    parser.add_argument('--plot_figures',
                        action='store_true',
                        help='Plot figures')
    parser.add_argument('--override',
                        nargs='+',
                        help='Override parameters in the config file')

    return parser.parse_args()


def collect_test_configs(
        args: Optional[argparse.Namespace] = None,
        wandb_run_dir: Optional[str] = None,
        model_ckpt_fname: Optional[str] = None,
        ) -> Dict:
    """
    Collect configurations from the config file from the wandb run directory and command line arguments for testing.

    Parameters:
    ----------
    - args (argparse.Namespace): The command line arguments.

    Returns:
    ----------
    - Dict: The configuration dictionary.
    """
    assert args is not None or wandb_run_dir is not None, \
        "Either args or wandb_run_dir must be provided"
    
    # get wandb run directory from command line arguments if provided, otherwise from the wandb_run_dir argument
    if args is not None:
        wandb_run_dir = args.wandb_run_dir
        
    # get config file path from wandb run directory
    config_fname = Path(wandb_run_dir) / 'files' / 'user_specified_config.yaml'
    
    # read parameters from config file
    with open(config_fname, "r") as f:
        config = yaml.safe_load(f)

    # set model checkpoint file name from command line arguments if provided, otherwise from the model_ckpt_fname argument, else find best checkpoint from wandb run directory
    if args is not None and args.model_ckpt_fname is not None:
        config['model']['model_ckpt_fname'] = args.model_ckpt_fname
    elif model_ckpt_fname is not None:
        config['model']['model_ckpt_fname'] = model_ckpt_fname
    else:
        config['model']['model_ckpt_fname'] = find_best_checkpoint(wandb_run_dir)

    # write wandb_run_directory and model_ckpt_fname to config
    config['experiment']['wandb_run_dir'] = wandb_run_dir
    config['model']['model_ckpt_fname'] = config['model']['model_ckpt_fname']
    config['experiment']['compute_metrics'] = args.compute_metrics if args is not None else False
    config['experiment']['plot_figures'] = args.plot_figures if args is not None else False

    return config


def find_best_checkpoint(
        wandb_run_dir: str,
        mode: Literal['min', 'max'] = 'max',
        metric_name: Literal['mmd_eigenvalues', 'pearson_1hop_nbr'] = 'pearson_1hop_nbr',
    ) -> str:
    """
    Find checkpoint file name with best metric value.

    Parameters:
    ----------
    - wandb_run_dir: str
        The directory containing the wandb run files.
    - mode: Literal['min', 'max']
        The mode to find the best checkpoint.
    - metric_name: str
        The name of the metric to find the best checkpoint.

    Returns:
    ----------
    - str: The path to the checkpoint file with the best metric value.
    - e.g. if mode is 'max' and metric_name is 'pearson_1hop_nbr', the function will return the checkpoint file with the highest Pearson correlation between the original and reconstructed cell-gene matrices
    - e.g. if mode is 'min' and metric_name is 'mmd_eigenvalues', the function will return the checkpoint file with the lowest MMD between the eigenvalue distributions of the original and reconstructed graphs
    """
    ckpt_dir = Path(wandb_run_dir) / 'files' / 'checkpoints'

    # initialize value of best metric to be the minimum or maximum possible value
    best_metric_val = np.inf if mode == 'min' else -np.inf
    best_ckpt = None
    for f in os.listdir(ckpt_dir):
        if f.endswith('.ckpt'):
            try:
                # Extract metric value from filename like "epoch=X-metric_name=Y.ckpt"
                metric_val = float(f.split(f'{metric_name}=')[1].split('.ckpt')[0])
                if (mode == 'min' and metric_val < best_metric_val) or (mode == 'max' and metric_val > best_metric_val):
                    best_metric_val = metric_val
                    best_ckpt = f
            except:
                continue
    print(f"Best checkpoint found: {best_ckpt}")

    return ckpt_dir / best_ckpt