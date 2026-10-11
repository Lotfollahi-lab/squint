# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Changed

-   Renamed the package from `vqniche` to `squint`: the source moved from `src/vqniche` to `src/squint`, imports are now `import squint` / `from squint... import ...`, and the distribution name in `pyproject.toml` is `squint`.
-   Renamed the model and encoder classes and their modules: `VQNiche` -> `Squint` (`squint.models.squint`), `VQNiche_Dual` -> `Squint_Dual` (`squint.models.squint_dual`), `VQNiche_Encoder` -> `Squint_Encoder` (`squint.encoders.squint_encoder`), `VQNiche_Dual_Encoder` -> `Squint_Dual_Encoder` (`squint.encoders.squint_dual_encoder`). The old names are still accepted as `model_name` / `encoder_name` values in configs and checkpoints (`squint.initializers.initialize.normalize_model_name`).
-   The squint-reproducibility train configs `*_vqniche_graphsage*.yaml` are now `*_squint_graphsage*.yaml`, and the conda environment is called `squint-reproducibility`.
-   Migrating an existing environment: `pip uninstall vqniche`, delete the leftover `src/vqniche` directory in your checkout, then `pip install -e .` (see the README).

### Fixed

-   `compute_ilisi` in `examples/compute_inference_metrics.py` permutes the cells (seeded) before the NNDescent neighbour search, so exact ties in quantized embeddings are broken at random instead of by storage order, which picked same-section neighbours and understated SQUINT's iLISI. It returns the scaled median of `scib_metrics.ilisi_knn` and skips iLISI with a message when `scib-metrics` cannot be imported, instead of falling back to an unscaled inverse-Simpson mean, which is a different statistic (89d9c82). The iLISI values of the camera-ready paper were recomputed with the corrected estimator (squint-reproducibility, `analysis/benchmarking/ilisi_recompute/`).

### Added

-   Basic tool, preprocessing and plotting functions
