# VQNiche

Tokenization for Tissue Sections

## Installation

This is tested with Python 3.10. To install the latest development version, first activate your Python Environment and install this package in editable mode as follows:

```bash
pip install -e .
```

## Usage

The model training pipeline consists of the following steps:
1. Build the dataset (see vqniche.dataset.in_memory_dataset_blob.py).
2. Initialize the DataModule (see vqniche.dataloaders.in_memory_datamodule.py).
3. Initialize the Model (see vqniche.models.vqgraph.py).
4. Initialize the Logger, and Checkpoints, and Trainer (see vqniche.utils.initialize.py).
5. Train the Model.


## Contact

For questions and issues, please reach out to am84@sanger.ac.uk.
