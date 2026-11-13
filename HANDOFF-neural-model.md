# Handoff: improving the neural constraint model

Written 2026-10-26 from a review of the last training run. The goal of the next session is to
make the model accept more valid poses without ever calling an invalid pose safe, and to use the
work as a hands-on way for Hen to learn how neural networks are trained: explain each change and
what it teaches, and compare metrics before and after.

Read first: [AGENTS.md](AGENTS.md), and the "External Neural Constraint Pipeline" and "Next Session"
sections of [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md). The live-solve caching work listed there
(Next Session step 1) is separate and still open; this handoff covers model quality only.

## What exists

- `tools/generate_neural_dataset.py` labels poses in background Blender: proxy geometry for most
  rows, full mesh for holdout and boundary rows.
- `tools/train_constraint_model.py`: `ConstraintModel` turns the pose `[tx, ty, tz, rx, ry, rz]`
  into 9 inputs (translation plus a 6D rotation), then two hidden layers of 64 with ReLU, then
  two heads:
  - validity: weighted binary cross-entropy
  - clearance in mm: softplus output, masked smooth-L1 loss
  Training uses Adam with weight decay, train-only normalization and leakage-safe group splits,
  keeps the best validation epoch, calibrates a conservative validity threshold on the full-mesh
  holdout, then exports to ONNX with a PyTorch/ONNX Runtime parity check.
- `tools/active_learning_loop.py` labels uncertain poses with the full mesh and retrains.

## Last run (training/, 2026-08-15)

The package in `training/` is fixed `L5`, moving `L4`, ±10 mm, ±45°. The run used sample budget
2000, 300 epochs, batch 32, hidden 64, learning rate 1e-3, weight decay 1e-4, and 6 active rounds
(up to 400 new samples). Dataset generation took about 6 s; training took about 37 s on CPU.

| Signal | Value | Reading |
|---|---|---|
| Train / validation loss at the end | 0.27 / 0.88 | Overfitting: validation was best at 0.57 and then rose |
| Holdout full-mesh labels | 21 valid, 278 invalid (299) | Strong class imbalance |
| Confusion (threshold 0.9967) | TP 8, FN 13, TN 278, FP 0 | Never false-safe, but rejects 62% of valid poses |
| Clearance MAE | 1.06 mm | About 2x the 0.5 mm resolution target |
| ONNX parity max diff | 3.8e-6 | Fine |

Caveat: these artifacts may predate later pipeline changes. The one-click default budget is now
10,000 poses, and DEVELOPMENT_PLAN's verified pair is sacrum/L5. Re-run the baseline before
comparing.

## Plan, in order (one change at a time; record metrics after each)

1. **Baseline.** Re-run the current pipeline on the L5/L4 package and save `metrics.json` and the
   loss curve as the reference.
   *Teaches:* reading train vs validation curves, confusion matrices, false-safe vs false-invalid.
2. **More data, concentrated near the boundary.** Data costs seconds, so raise the budget to
   10k–50k and give the balancing and refinement stages a larger share.
   *Teaches:* data volume vs generalization; this is the most likely big win.
3. **Class balance.** Weight the loss by class (`pos_weight`) or sample batches evenly; make sure
   the holdout has enough valid poses (aim for 200 or more) so the threshold rests on evidence.
   *Teaches:* imbalanced classification and calibration.
4. **Regularization and early stopping.** Stop when validation stops improving (patience);
   try dropout or stronger weight decay; compare the curves.
   *Teaches:* overfitting control.
5. **Input encoding.** Add Fourier features of the normalized pose (sin/cos at several
   frequencies) so a small MLP can follow sharp collision boundaries; also try width 128 and
   3 layers.
   *Teaches:* spectral bias and feature encoding (the same idea as NeRF).
6. **Problem framing (experiment).** Predict signed clearance (negative means penetration) as
   the main output and derive validity as clearance ≥ 0.5 mm, so the two heads can't disagree.
   This needs signed labels from the dataset generator: today softplus forces clearance ≥ 0 and
   overlap is masked.
   *Teaches:* choosing the target and loss to fit the physics.

## Rules to keep

- Never accept a model with a false-safe rate above 0 on the full-mesh holdout. Improve recall
  without losing that.
- Keep every manifest, compatibility and parity check; `model_manifest.json` is still written last.
- Run training outside Blender's UI thread (`tools/run_neural_training.ps1`), and run
  `python -m compileall biomechanics_addon` plus the host tests after code changes.
- Results describe geometric feasibility under the label rule, not anatomy.

## Done looks like

On the full-mesh holdout: recall on valid poses clearly above 38% (8/21) with zero false-safe
results; clearance MAE at or below 0.5 mm; and a short table in this file comparing the baseline
with each step.
