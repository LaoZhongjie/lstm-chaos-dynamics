# Edge of Chaos in Deep Recurrent Networks (Ongoing Research)

This repository contains an ongoing research pipeline for studying whether the **generalization optimum** of deep recurrent networks emerges near the **critical dynamical regime** (the *edge of chaos*).

## Research Goal

Primary question:

> Does the best generalization point in deep recurrent training align with a transition region between ordered and chaotic dynamics?

Current hypothesis:

- The epoch with minimum test loss is likely to lie near a dynamical critical region, rather than deep inside purely ordered or strongly chaotic regimes.

This project is **work in progress**. No final scientific claim is made yet.

## Method Summary

For each training epoch, we study both optimization behavior and hidden-state dynamics:

1. Train an LSTM classifier on IMDB sentiment.
2. Save epoch checkpoints.
3. Analyze hidden-state dynamics on a fixed analysis subset:
   - **FTLE (Benettin method)** as a local sensitivity indicator.
   - **Bifurcation-style trajectory statistics** from perturbed hidden-state evolution.
4. Compare:
   - test loss minima / generalization behavior
   - FTLE trends
   - bifurcation-map transitions

## Current Pipeline

Main entrypoint:

```bash
python main.py
```

Useful flags:

```bash
# quick validation run
python main.py -qt

# custom training/analysis horizons
python main.py -te 500 -ae 500

# skip completed stages
python main.py -st      # skip training
python main.py -sa      # skip analysis
python main.py -sv      # skip visualization
```

## Project Structure

```text
.
├── main.py                  # End-to-end runner (train -> analyze -> visualize)
├── config.py                # Global experiment configuration
├── train.py                 # LSTM training pipeline
├── model.py                 # LSTM model
├── data_loader.py           # IMDB data loading/preprocessing
├── analysis_runner.py       # Epoch-wise chaos analysis orchestrator
├── asymptotic_analyzer.py   # Hidden-state perturbation / trajectory analysis
├── ftle_analyzer.py         # Benettin FTLE estimator
├── visualize_results.py     # Research plots and animation generation
├── seed_utils.py            # Hierarchical seed manager for reproducibility
└── README.md
```

## Dataset

- IMDB Large Movie Review Dataset (binary sentiment)
- Source: <https://huggingface.co/datasets/stanfordnlp/imdb>

## Reproducibility

- Seed control is centralized via `HierarchicalSeedManager`.
- Analysis uses a fixed sampled subset to reduce stochastic variance across epochs.

## Outputs

Typical generated artifacts:

```text
results/
├── training_history.json
├── analysis_summary.json
├── chaos_analysis_results.h5
└── figures/
    ├── training_curves.png
    ├── test_loss_with_ftle.png
    └── combined_bifurcation_animation_gpu.mov

checkpoints/
└── model_epoch_*.pt
```

## Interpretation Note

This codebase provides measurement and visualization tools.  
Interpreting order/chaos transitions and linking them to generalization optima remains an active research task.

## Project Status

- Active, ongoing research
- Code and analysis protocol may change as hypotheses are refined
- Planned additions include stronger statistical tests and multi-seed robustness analysis
