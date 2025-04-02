# VQNiche

Tokenization for Tissue Sections

## Installation

You need to have Python 3.10 or newer installed on your system.

To install the latest development version, first activate your Python Environment and `cd` to the directory where this repository is stored. Then, execute the following:

```bash
pip install -e .
```

## Usage

The model trainingpipeline consists of the following steps:
1. Build the dataset (see vqniche.dataset.in_memory_dataset_blob.py).
2. Initialize the DataModule (see vqniche.dataloaders.in_memory_datamodule.py).
3. Initialize the Model (see vqniche.models.vqgraph.py).
4. Initialize the Logger, and Checkpoints, and Trainer (see vqniche.utils.initialize.py).
5. Train the Model.


## Contact

For questions and help requests, you can reach out to am84@sanger.ac.uk or ls34@sanger.ac.uk.
