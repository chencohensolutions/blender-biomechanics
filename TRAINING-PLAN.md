# Blender Biomechanics: training plan (6 weeks, ~12–15 h/week)

Goal: turn the project into honest, public evidence of **training neural networks** — tracked experiments, real metrics against a baseline, image models, tuning, compression and a clean pipeline — so the CV can say what ML job ads ask for (see `nn-training-path.md`, gaps 4, 5, 7, 8).

Repo: `C:\src\blender-biomechanics` (GitHub: chencohensolutions/blender-biomechanics).
Read first in the session: `AGENTS.md`, `HANDOFF-neural-model.md`, the "External Neural Constraint Pipeline" and "Next Session" sections of `DEVELOPMENT_PLAN.md`.

## How to work in the dedicated session

- **Learning mode.** Hen does the work with Claude as a coach: before each change, explain what it is and what it should teach; after it, compare metrics before vs after and explain the result. Hen should be able to explain every step in an interview.
- **One change at a time.** Every change gets a run, a saved `metrics.json`, and a line in `training/EXPERIMENTS.md` (date, change, metrics, what we learned, keep/revert).
- **Never false-safe.** Validity must never call an invalid pose safe on the full-mesh holdout (FP = 0). Everything else is optimized under that rule.
- **Small commits on a branch**, a PR per phase. Tests stay green (`tests/`).
- **No CV claims until the numbers exist.** Each phase below has a CV line with blanks; fill them only with real results.

## Phase 0 — Baseline (1–2 sessions)

- Re-run the current pipeline on the L5/L4 package (or sacrum/L5 if that's the verified pair now). Save `metrics.json`, loss curves, confusion matrix as the **reference run**.
- Add Weights & Biases (free) or MLflow logging, config files (YAML), fixed seeds, a frozen and versioned train/val/holdout split, one-command training.
- Done when: one command reproduces the baseline and it shows up in the tracking dashboard.
- Teaches: reading train vs validation curves; false-safe vs false-invalid; reproducibility.

## Phase 1 — Fix the constraint model (week 1–2) — the HANDOFF steps

Last known run: overfits (val loss 0.57 → 0.88), holdout 21 valid vs 278 invalid, rejects 62% of valid poses, clearance MAE 1.06 mm (target 0.5 mm).

1. More data near the boundary: 10k–50k samples, bigger share for balancing and refinement stages.
2. Class balance: `pos_weight` or balanced batches; holdout with 200+ valid poses so the threshold has evidence.
3. Regularization: early stopping with patience, dropout / weight decay; compare curves.
4. Input encoding and capacity: Fourier features of the pose; width 128, 3 layers.
5. Re-calibrate the conservative threshold, re-export ONNX, re-check parity and Blender inference.

- Targets: FP = 0; valid-pose acceptance from 38% to ≥ 80%; clearance MAE ≤ 0.5 mm.
- CV line: "Trained a two-head PyTorch model (validity + clearance) on N synthetic poses labeled in Blender; raised valid-pose acceptance from 38% to X% with zero false-safe predictions and cut clearance error from 1.06 to Y mm (class weighting, boundary sampling, Fourier features, early stopping)."

## Phase 2 — Image models on Blender renders (week 3)

- Render images of each pose (RGB + depth) with labels (feasible, clearance mm). Keep the same split by pose group.
- Fine-tune **ResNet-50** and **DINOv2 ViT-S** (`timm` / Hugging Face). Compare: frozen backbone + linear probe, full fine-tune, LoRA. Report accuracy, MAE, calibration.
- GPU: Kaggle free tier (30 h/week) or Colab.
- Teaches: transfer learning, CNN vs ViT, fine-tuning strategies, data loaders and augmentation for images.
- CV line: "Fine-tuned ResNet-50 and DINOv2 ViT on N Blender-rendered images; compared linear probe, full fine-tune and LoRA (best: X% / Y mm MAE)."

## Phase 3 — Data strategy and tuning (week 4)

- **Optuna** sweep (learning rate, weight decay, width, augmentation strength), logged to the tracker.
- **Domain randomization** in Blender (lighting, textures, camera) for the image model.
- **Active learning vs random sampling** at equal label budget: plot a learning curve.
- Teaches: hyperparameter search, sampling strategy, augmentation — the exact words in ML job ads.
- CV line: "Ran an N-trial Optuna search and showed active learning reaches the same accuracy with X% fewer labels than random sampling."

## Phase 4 — Compression and deployment (week 5)

- Quantize the models (ONNX Runtime dynamic/static int8; optional pruning). Measure size, latency on CPU inside Blender, accuracy drop. Keep the ONNX parity checks.
- Teaches: model compression trade-offs, inference performance.
- CV line: "Quantized the models to int8 ONNX: Xx smaller, Y ms per inference on CPU in Blender, accuracy within Z%."

## Phase 5 — Pipeline as software (week 6)

- Dockerfile for training; GitHub Actions CI that runs tests + a tiny training smoke run + an **evaluation gate** (CI fails if holdout metrics regress or any false-safe appears).
- Dataset versioning (DVC or hashed manifests) and a model card per released model.
- Teaches: MLOps basics (CI/CD for ML, reproducibility, model registry thinking).
- CV line: "Built CI for model training (Docker, GitHub Actions) with an evaluation gate that blocks regressions; versioned datasets and model cards."

## Phase 6 — Publish (end of week 6)

- README "Results" section: table of experiments (baseline → final), plots, how to reproduce in one command.
- Upload the final models + model cards to Hugging Face.
- Short write-up (blog post or `docs/RESULTS.md`): what worked, what didn't, why.
- Update `master-cv.md` Blender Biomechanics bullets with the real numbers.

## Later (separate repo, from nn-training-path.md weeks 4–8)

3D U-Net CT vertebra segmentation on public VerSe/TotalSegmentator data with MONAI, benchmark vs nnU-Net, multi-GPU DDP on a rented machine (Vast.ai/RunPod, ~$50–100 total), then plug the model back into this project's CT-to-mesh pipeline.
