import os
import yaml
import argparse
from typing import Dict
from pathlib import Path


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
    parser.add_argument('--model_ckpt',
                        type=str,
                        default=None,
                        help='Optional model checkpoint file name')
    parser.add_argument('--override',
                        nargs='+',
                        help='Override parameters in the config file')

    return parser.parse_args()


def collect_test_configs(args: argparse.Namespace) -> Dict:
    """
    Collect configurations from the config file from the wandb run directory and command line arguments for testing.

    Parameters:
    ----------
    - args (argparse.Namespace): The command line arguments.

    Returns:
    ----------
    - Dict: The configuration dictionary.
    """
    # Get the config file name from command line argument
    config_fname = Path(args.wandb_run_dir) / 'files' / 'config.yaml'

    # Read parameters from config file
    with open(config_fname, "r") as f:
        config = yaml.safe_load(f)

    # Set model checkpoint file name
    config['experiment']['wandb_run_dir'] = args.wandb_run_dir
    if args.model_ckpt:
        config['model']['model_ckpt'] = args.model_ckpt
    else:
        config['model']['model_ckpt'] = find_best_checkpoint(args.wandb_run_dir)

    return config


def find_best_checkpoint(wandb_run_dir):
    """
    Find checkpoint file name with highest validation accuracy.

    Parameters:
    ----------
    - wandb_run_dir: str
        The directory containing the wandb run files.

    Returns:
    ----------
    - str: The path to the checkpoint file with the highest validation accuracy.
    """
    ckpt_dir = Path(wandb_run_dir) / 'files' / 'checkpoints'

    best_acc = -1
    best_ckpt = None
    for f in os.listdir(ckpt_dir):
        if f.endswith('.ckpt'):
            try:
                # Extract val_acc from filename like "epoch=X-val_acc=Y.ckpt"
                val_acc = float(f.split('val_acc=')[1].split('.ckpt')[0])
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_ckpt = f
            except:
                continue
    print(f"Best checkpoint found: {best_ckpt}")

    return ckpt_dir / best_ckpt