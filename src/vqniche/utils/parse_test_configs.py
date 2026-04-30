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
                        default=None,
                        help='Metric name to find the best checkpoint')
    parser.add_argument('--mode',
                        type=str,
                        default=None,
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

    # get config file path. Two supported layouts:
    #   (a) Flat run dir produced by examples/run_squint_mmb_smb.py:
    #         <run_dir>/user_specified_config.yaml
    #   (b) Legacy wandb-controlled layout:
    #         <run_dir>/files/user_specified_config.yaml
    flat_cfg = Path(wandb_run_dir) / 'user_specified_config.yaml'
    legacy_cfg = Path(wandb_run_dir) / 'files' / 'user_specified_config.yaml'
    if flat_cfg.exists():
        config_fname = flat_cfg
    elif legacy_cfg.exists():
        config_fname = legacy_cfg
    else:
        raise FileNotFoundError(
            f"Could not find user_specified_config.yaml under {wandb_run_dir} "
            f"(checked both {flat_cfg} and {legacy_cfg})."
        )
    
    # read parameters from config file
    with open(config_fname, "r") as f:
        config = yaml.safe_load(f)

    # set model checkpoint file name from command line arguments if provided, otherwise from the model_ckpt_fname argument, else find best checkpoint from wandb run directory
    if args is not None:
        if args.model_ckpt_fname is not None:
            config['model']['model_ckpt_fname'] = args.model_ckpt_fname
        else:
            if args.metric_name is not None:
                metric_name = args.metric_name
            else:
                metric_name = config['trainer']['monitor']
            if args.mode is not None:
                mode = args.mode
            else:
                mode = config['trainer']['checkpoint_params']['mode']
            config['model']['model_ckpt_fname'] = find_best_checkpoint(
                                                        wandb_run_dir,
                                                        mode=mode,
                                                        metric_name=metric_name,
                                                    )
    else:
        if model_ckpt_fname is not None:
            config['model']['model_ckpt_fname'] = model_ckpt_fname
        else:
            mode = config['trainer']['checkpoint_params']['mode']
            metric_name = config['trainer']['monitor']
            config['model']['model_ckpt_fname'] = find_best_checkpoint(
                                                        wandb_run_dir,
                                                        mode=mode,
                                                        metric_name=metric_name,
                                                    )

    # write wandb_run_directory to config (model_ckpt_fname was already set above)
    config['experiment']['wandb_run_dir'] = wandb_run_dir
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
    # Two supported layouts (mirrors collect_test_configs):
    #   (a) Flat run dir:   <run_dir>/checkpoints/
    #   (b) Legacy wandb:   <run_dir>/files/checkpoints/
    flat_ckpt_dir = Path(wandb_run_dir) / 'checkpoints'
    legacy_ckpt_dir = Path(wandb_run_dir) / 'files' / 'checkpoints'
    if flat_ckpt_dir.exists():
        ckpt_dir = flat_ckpt_dir
    elif legacy_ckpt_dir.exists():
        ckpt_dir = legacy_ckpt_dir
    else:
        raise FileNotFoundError(
            f"Could not find checkpoints directory under {wandb_run_dir} "
            f"(checked both {flat_ckpt_dir} and {legacy_ckpt_dir})."
        )

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
            except (IndexError, ValueError):
                continue

    if best_ckpt is None:
        # Common cause: trainer.monitor was changed post-training, or only
        # `last.ckpt` exists (which doesn't carry the metric in its filename).
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
        raise FileNotFoundError(
            f"No checkpoint in {ckpt_dir} matched the metric '{metric_name}'. "
            f"Found {len(ckpt_files)} ckpt(s): {ckpt_files[:5]}"
            f"{' ...' if len(ckpt_files) > 5 else ''}"
        )
    print(f"Best checkpoint found: {best_ckpt}")

    return ckpt_dir / best_ckpt