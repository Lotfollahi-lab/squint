#!/usr/bin/env python
"""
Train the SQUINT stage-2 spatial code prior on a frozen ``predicted_adata.h5ad``.

The stage-2 model (``vqniche.stage2``) is a graph-native, MaskGIT-style masked
code transformer: it predicts held-out cells' discrete code stacks from the
spatial context of observed cells. It NEVER reads expression and NEVER touches
the stage-1 (SQUINT VQ-VAE) training path -- it consumes only the codes +
positions exported in ``predicted_adata.h5ad``.

What this script does:
  1. read the frozen codes / positions / sections (AnnDataCodeSource)
  2. build a Stage2Config from the run's actual codebook sizes
  3. train with PyTorch Lightning (masked cross-entropy; contiguous block masks)
  4. save weights + config (vqniche.stage2.checkpoint)
  5. run a region-holdout in-painting evaluation (decoded top-1 code accuracy)

Example (smoke test on the mmb/smb run):
  python examples/run_stage2.py \
      --predicted-adata /nfs/team361/sb75/squint-reproducibility/artifacts/\
mmb0-1b_smb1-1b_1p/<variant>/<TS>_seed0/predicted_adata.h5ad \
      --smoke

Full run: drop --smoke and tune --max-steps / --d-model / --patch-size.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Make `import vqniche` work whether or not the package is pip-installed.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predicted-adata", required=True,
                   help="Path to a frozen SQUINT predicted_adata.h5ad.")
    p.add_argument("--out-dir", default=None,
                   help="Output dir (default: <run_dir>/stage2/<timestamp>).")

    # data / patching
    p.add_argument("--patch-size", type=int, default=1024)
    p.add_argument("--knn", type=int, default=16)
    p.add_argument("--mask-kind", choices=["block", "random"], default="block")
    p.add_argument("--mask-frac-min", type=float, default=0.10)
    p.add_argument("--mask-frac-max", type=float, default=0.50)
    p.add_argument("--oversample", type=float, default=1.0)
    p.add_argument("--coord-key", default="spatial")
    p.add_argument("--section-key", default="adata_batch_id")

    # model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pos-num-freqs", type=int, default=64)
    p.add_argument("--no-hierarchical", action="store_true",
                   help="Disable cell->niche conditioning (branches independent).")
    # backbone architecture (the stage-2 ablation axis; same embedding/heads/decode)
    p.add_argument("--arch", default="transformer",
                   choices=["transformer", "gnn", "labelprop", "graphmae", "gps",
                            "diffusion"],
                   help="Stage-2 backbone: transformer (default) | gnn | labelprop "
                        "| graphmae | gps | diffusion.")
    p.add_argument("--graph-knn", type=int, default=16,
                   help="Neighbours/cell for message-passing archs (gnn/gps/labelprop/graphmae).")
    p.add_argument("--gnn-aggregator", default="mean", choices=["mean", "sum", "max"])
    p.add_argument("--prop-steps", type=int, default=10,
                   help="labelprop: APPNP propagation rounds.")
    p.add_argument("--prop-alpha", type=float, default=0.1,
                   help="labelprop: APPNP teleport probability.")

    # optimisation
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--l0-weight", type=float, default=1.0,
                   help="Ablation: up-weight the L0 (coarse) code CE vs L1.")
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--val-check-interval", type=int, default=1000)

    # decoding (eval)
    p.add_argument("--decode-steps", type=int, default=12)
    p.add_argument("--eval-regions", type=int, default=8)
    p.add_argument("--eval-holdout-frac", type=float, default=0.1)
    p.add_argument("--skip-eval", action="store_true")

    # TRUE data-split holdout: in-paint exactly the cells SQUINT held out
    # (obs[holdout-key] == holdout-split) instead of random synthetic holes.
    # Held-out cells are then EXCLUDED from stage-2 training/val patches.
    p.add_argument("--holdout-split", default=None,
                   help="obs value marking held-out cells (e.g. 'test'). "
                        "Enables the true region-holdout train/eval protocol.")
    p.add_argument("--holdout-key", default="data_split",
                   help="obs column carrying the train/test split (default data_split).")
    p.add_argument("--eval-chunk-size", type=int, default=0,
                   help="Max held-out cells per in-painting chunk (0 -> patch_size//2).")

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", default="32-true",
                   help="PL precision, e.g. '32-true', '16-mixed', 'bf16-mixed'.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny/fast config for an end-to-end sanity run.")
    return p.parse_args(argv)


def build_config(args, source):
    from vqniche.stage2.config import (
        Stage2Config, DataConfig, ModelConfig, DecodeConfig, OptimConfig,
        default_branches,
    )

    sizes = source.branch_sizes()
    branches = default_branches(sizes["cell"], sizes["niche"])

    if args.smoke:
        data = DataConfig(patch_size=min(256, args.patch_size), knn=8,
                          mask_kind=args.mask_kind, mask_frac_min=0.2,
                          mask_frac_max=0.4, oversample=args.oversample,
                          coord_key=args.coord_key, section_key=args.section_key)
        model = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128,
                            dropout=args.dropout, pos_num_freqs=16,
                            hierarchical=not args.no_hierarchical,
                            arch=args.arch, graph_knn=args.graph_knn,
                            gnn_aggregator=args.gnn_aggregator,
                            prop_steps=args.prop_steps, prop_alpha=args.prop_alpha)
        optim = OptimConfig(lr=args.lr, weight_decay=args.weight_decay,
                            warmup_steps=20, max_steps=80, grad_clip=args.grad_clip,
                            batch_size=min(4, args.batch_size), l0_weight=args.l0_weight)
        decode = DecodeConfig(steps=6)
        eval_regions = min(3, args.eval_regions)
    else:
        data = DataConfig(patch_size=args.patch_size, knn=args.knn,
                          mask_kind=args.mask_kind, mask_frac_min=args.mask_frac_min,
                          mask_frac_max=args.mask_frac_max, oversample=args.oversample,
                          coord_key=args.coord_key, section_key=args.section_key)
        model = ModelConfig(d_model=args.d_model, n_layers=args.n_layers,
                            n_heads=args.n_heads, d_ff=args.d_ff, dropout=args.dropout,
                            pos_num_freqs=args.pos_num_freqs,
                            hierarchical=not args.no_hierarchical,
                            arch=args.arch, graph_knn=args.graph_knn,
                            gnn_aggregator=args.gnn_aggregator,
                            prop_steps=args.prop_steps, prop_alpha=args.prop_alpha)
        optim = OptimConfig(lr=args.lr, weight_decay=args.weight_decay,
                            warmup_steps=args.warmup_steps, max_steps=args.max_steps,
                            grad_clip=args.grad_clip, batch_size=args.batch_size,
                            l0_weight=args.l0_weight)
        decode = DecodeConfig(steps=args.decode_steps)
        eval_regions = args.eval_regions

    cfg = Stage2Config(branches=branches, data=data, model=model, decode=decode,
                       optim=optim, seed=args.seed)
    return cfg, eval_regions


def main(argv=None):
    args = parse_args(argv)

    import torch
    import pytorch_lightning as pl
    from vqniche.stage2.data import AnnDataCodeSource
    from vqniche.stage2.datamodule import Stage2DataModule
    from vqniche.stage2.lightning import Stage2LightningModule
    from vqniche.stage2.checkpoint import save_stage2

    pl.seed_everything(args.seed, workers=True)

    # ---- output dir -------------------------------------------------------
    if args.out_dir:
        out_dir = args.out_dir
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.dirname(os.path.abspath(args.predicted_adata))
        out_dir = os.path.join(run_dir, "stage2", ts)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[stage2] out_dir: {out_dir}")

    # ---- frozen source ----------------------------------------------------
    print(f"[stage2] reading codes from {args.predicted_adata}")
    source = AnnDataCodeSource(
        args.predicted_adata, coord_key=args.coord_key, section_key=args.section_key,
        holdout_key=(args.holdout_key if args.holdout_split else None),
        holdout_value=(args.holdout_split if args.holdout_split else "test"),
    )
    print(f"[stage2] {source.n_cells} cells, {source.section_ids.size} section(s), "
          f"codebook sizes {source.branch_sizes()}")
    use_split_eval = bool(args.holdout_split) and source.has_holdout
    if args.holdout_split and not source.has_holdout:
        print(f"[stage2] WARNING: no cells with obs['{args.holdout_key}']=="
              f"'{args.holdout_split}' -- falling back to random-hole eval, "
              f"training on the full tissue.")
    elif use_split_eval:
        n_hold = int(source.holdout_mask.sum())
        print(f"[stage2] data-split holdout ON: {n_hold} held-out cells "
              f"(obs['{args.holdout_key}']=='{args.holdout_split}') excluded "
              f"from training; in-painted at eval.")

    # ---- config -----------------------------------------------------------
    cfg, eval_regions = build_config(args, source)
    with open(os.path.join(out_dir, "stage2_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"[stage2] targets: {cfg.prediction_targets}")

    # ---- data + model -----------------------------------------------------
    dm = Stage2DataModule(source, cfg, num_workers=args.num_workers,
                          restrict_train_to_split=use_split_eval)
    lit = Stage2LightningModule(cfg)
    n_params = sum(p.numel() for p in lit.model.parameters())
    print(f"[stage2] model params: {n_params/1e6:.2f}M")

    # ---- trainer ----------------------------------------------------------
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=out_dir, filename="stage2_last", save_last=True,
        every_n_train_steps=max(1, args.val_check_interval),
    )
    lr_cb = pl.callbacks.LearningRateMonitor(logging_interval="step")
    csv_logger = pl.loggers.CSVLogger(save_dir=out_dir, name="logs")

    trainer = pl.Trainer(
        max_steps=cfg.optim.max_steps,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        logger=csv_logger,
        callbacks=[ckpt_cb, lr_cb],
        val_check_interval=min(args.val_check_interval, cfg.optim.max_steps),
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        enable_progress_bar=True,
        gradient_clip_val=None,    # handled in the module (on_before_optimizer_step)
    )

    print("[stage2] training...")
    trainer.fit(lit, dm.train_dataloader(), dm.val_dataloader())

    # ---- save -------------------------------------------------------------
    save_stage2(out_dir, lit.model, cfg)
    print(f"[stage2] saved model + config to {out_dir}")

    # ---- in-painting eval -------------------------------------------------
    if not args.skip_eval:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lit.model.to(device)
        if use_split_eval:
            # in-paint exactly the cells SQUINT held out (data_split == test)
            from vqniche.stage2.evaluate import holdout_split_eval

            chunk = args.eval_chunk_size or max(64, cfg.data.patch_size // 2)
            if args.smoke:
                chunk = min(chunk, 64)
            summary = holdout_split_eval(
                lit.model, source, cfg, chunk_size=chunk, device=device,
                out_csv=os.path.join(out_dir, "stage2_eval.csv"),
                codes_out=os.path.join(out_dir, "stage2_predicted_codes.npz"),
                save_soft=True,   # also persist soft code distributions for --decode-mode soft
            )
            drop_key = "per_chunk"
        else:
            from vqniche.stage2.evaluate import region_holdout_eval

            summary = region_holdout_eval(
                lit.model, source, cfg,
                n_regions=eval_regions, holdout_frac=args.eval_holdout_frac,
                seed=args.seed, device=device,
                out_csv=os.path.join(out_dir, "stage2_eval.csv"),
            )
            drop_key = "per_region"
        with open(os.path.join(out_dir, "stage2_eval.json"), "w") as f:
            json.dump({k: v for k, v in summary.items() if k != drop_key},
                      f, indent=2)
    print("[stage2] DONE.")
    return out_dir


if __name__ == "__main__":
    main()
