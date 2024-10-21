## Contributing:

- Branches:
    - All development should happen in a branch.
    - Define a limited scope and create a branch from `dev`.
    - Name the branch according to the following convention: prefix such as `feature/`,  `bugfix/`, `docs/`, `tests/`, `refactor/` and name such as `name-of-feature`.
    - Commit frequently and create a pull request once the branch is ready to be merged.
- Notebooks:
    - Place notebooks in `src/vqniche/notebooks`.
    - Create this folder if it does not exist.
- Related Repositories:
    - Code for our method(s) goes into `vqniche` repository.
    - Code for reproducibility goes into `vqniche-reproducibility` repository.

## Building a Clean Conda Environment:

Name of Environment: `vqniche-reproducibility`
Install packages in the following order to manage package dependencies and avoid conflicts:
1. Grab a small interactive CPU node session:
    - `bsub -q normal -n4 -M25G -R"select[mem>25G] rusage[mem=25G]" -Is /bin/bash`
2. Create a blank new environment:
    - `conda create --name vqniche-reproducibility python=3.10`
    - `conda activate vqniche-reproducibility`
3. Add scanpy
    - `pip install scanpy`
4. Add pytorch (version 2.2) for Cuda 12.1
    - `pip install torch==2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
5. Add Pytorch Lightning
    - `pip install lightning`
6. Add dgl for Torch 2.2
    - `pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html`
7. Add dependencies for pytorch geometric for Torch 2.2
    - `pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html`
8. Add pytorch geometric
    - `pip install torch_geometric`
9. Add squidpy:
    - `pip install squidpy`
10. Add datasets from Hugging Face:
    - `pip install datasets`
11. Add transformers from Hugging Face
    - `pip install transformers`
12. Add ipykernel
    - `pip install ipykernel`
13. Add einops
    - `pip install einops`
14. Add googledrivedownloader
    - `pip install googledrivedownloader`
15. Add category_encoders
    - `pip install category_encoders`
16. Add anndata
    - `pip install anndata`
17. Add wandb
    - `pip install wandb`
18. Add pre-commit
    - `pip install pre-commit`
19. `vqniche` package in editable mode:
    - `pip install -e .`
    - **Usage**: The command syntax is `pip install -e path/to/PackageDirectory`, where `path/to/PackageDirectory` is the path to the directory containing the package's `pyproject.toml` file. This means navigating to the root directory of the `vqniche` repository.