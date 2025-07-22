from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_logged_values_vs_epoch(
        df: pd.DataFrame,
        value_col: str = "Value",
        name_col: str = "Metric",
        mode_col: str = None,
        save_fname: Optional[str] = None,
        title: Optional[str] = None,
    ):
    """
    Plots each metric/loss in its own subplot vs epoch.
    If mode_col is provided, uses it as hue.
    """
    if df is None or df.empty:
        print("No data to plot!")
        return

    plot_names = df[name_col].unique()
    n_plots = len(plot_names)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, plot_name in enumerate(plot_names):
        ax = axes_flat[i]
        plot_data = df[df[name_col] == plot_name]
        if mode_col and mode_col in plot_data.columns:
            for mode in plot_data[mode_col].unique():
                mode_data = plot_data[plot_data[mode_col] == mode]
                ax.plot(mode_data['epoch'], mode_data[value_col], label=mode, marker='o', markersize=4)
            ax.legend()
        else:
            ax.plot(plot_data['epoch'], plot_data[value_col], marker='o', markersize=4, color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(plot_name)
        ax.set_title(plot_name)
        ax.grid(True, alpha=0.3)
    for i in range(n_plots, len(axes_flat)):
        axes_flat[i].set_visible(False)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_fname:
        plt.savefig(save_fname, dpi=300, bbox_inches='tight')
    plt.show()    


def read_on_train_epoch_end_logs(
        file_path: Path = None,
    ) -> pd.DataFrame:
    """
    This function reads the on_train_epoch_end_logs.csv file from the wandb run directory and returns it as a Pandas DataFrame.
    
    Parameters
    ----------
    file_path : str | Path
        Path to the on_train_epoch_end_logs.csv file.
        
    Returns
    -------
    df: pd.DataFrame
        DataFrame containing all loss terms and metrics logged during each training epoch.
        
    Notes:
    ------
    - This expects that the log file is named 'on_train_epoch_end_logs.csv' and is located in the wandb run directory.
    - The log file is expected to have loss terms and metrics as headings and their corresponding values as rows.
    """
    df = pd.read_csv(file_path)

    # --- Loss DataFrame ---
    # Identify loss columns -- those that contain 'loss' in their name and are not 'epoch'
    loss_columns = [col for col in df.columns if 'loss' in col.lower() and col != 'epoch']
    
    # For loss columns, create train/val pairs and add mode column
    if loss_columns:
        # Melt the dataframe to have a 'Mode' column for 'Train' and 'Val'
        id_vars = ['epoch']
        value_vars = loss_columns
        
        df_loss = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='Loss Term', 
            value_name='Value'
        )
        
        # Create 'Mode' column based on whether 'train' or 'val' is in the column name
        df_loss['Mode'] = df_loss['Loss Term'].apply(
            lambda x: 'Train' if 'train' in x.lower() else 'Val' if 'val' in x.lower() else 'Other'
        )
        
        # Simplify 'Metric' column by removing train_/val_ prefixes
        df_loss['Loss Term'] = df_loss['Loss Term'].apply(
            lambda x: x.replace('train_', '').replace('val_', '')
        )
    else:
        df_loss = None

    # --- Metrics DataFrame ---
    # Identify metric columns -- those that do not contain 'loss' in their name and are not 'epoch'
    metric_columns = [col for col in df.columns if col != 'epoch' and 'loss' not in col.lower()]

    if metric_columns:
        df_metrics = df.melt(
            id_vars=['epoch'],
            value_vars=metric_columns,
            var_name='Metric',
            value_name='Value'
        )
        
        # Create Mode column based on metric name
        def determine_mode(metric_name):
            if 'G_' not in metric_name:
                return 'Joint'
            elif 'G_hat_' in metric_name:
                return 'Reconstructed'
            else:
                return 'Original'
        
        df_metrics['Mode'] = df_metrics['Metric'].apply(determine_mode)
        
        # Remove G_hat_ and G_ prefixes from metric names
        df_metrics['Metric'] = df_metrics['Metric'].apply(
            lambda x: x.replace('G_hat_', '').replace('G_', '')
        )
    else:
        df_metrics = None

    return df_loss, df_metrics