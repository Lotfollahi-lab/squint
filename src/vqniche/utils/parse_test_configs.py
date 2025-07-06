import os
import yaml
import argparse
from typing import Dict, Optional
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
    config_fname = Path(wandb_run_dir) / 'files' / 'config.yaml'
    
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