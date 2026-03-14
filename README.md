# RNN Chaos Analysis Experiment

This repository implements chaos analysis for RNN models, calculating asymptotic distances to study training dynamics.

## Overview

This code analyzes the dynamical behavior of RNN models during training by:
1. Training an RNN on a classification task
2. Computing asymptotic distances at different training epochs
3. Visualizing training curves and chaos metrics

**Note**: Pattern detection (multiple descents, order-chaos transitions) requires manual inspection of the generated plots.

## Repository Structure

```
├── config.py                 # Configuration and hyperparameters
├── data_loader.py            # Data loading and preprocessing  
├── model.py                  # RNN model definition
├── chaos_analysis.py         # Asymptotic stability analysis
├── train.py                  # Training script
├── analyze_chaos.py          # Chaos analysis script
├── visualize_results.py      # Visualization script
├── run_experiment.py         # Complete pipeline
└── README.md                # This file
```

## Dataset

Uses the IMDB Large Movie Review Dataset (50,000 reviews):
- **Hugging Face**: https://huggingface.co/datasets/stanfordnlp/imdb
- Automatically downloaded via `datasets` library

## Installation

```bash
pip install -r requirements.txt
```

Configure in `config.py`:
- Set `DEVICE = 'cuda'` if GPU available
- Adjust `MAX_EPOCHS` as needed
- Modify paths if necessary

## Quick Start

### Full Pipeline
```bash
python run_experiment.py
```

### Quick Test
```bash
python run_experiment.py --quick-test
```

### Individual Steps

1. **Train model**:
```bash
python train.py
```

2. **Analyze chaos**:
```bash
python analyze_chaos.py
```

3. **Generate plots**:
```bash
python visualize_results.py
```

## Model Architecture

| Layer | Output Dimension |
|-------|-----------------|
| Embedding | 32 |
| RNN | 60 |
| Fully Connected | 1 |

**Training:**
- Optimizer: Adam (lr=0.0001)
- Loss: Binary Cross Entropy  
- Gradient clipping: 1.0
- Batch Size: 32

## Chaos Analysis Method

The asymptotic stability analysis:

1. Process input through RNN (first 500 timesteps)
2. Add small perturbation to hidden state
3. Continue both trajectories with zero inputs (timesteps 500-1599)
4. Calculate distance between original and perturbed trajectories
5. Compute geometric mean across test samples

**Interpretation** (requires manual inspection):
- **Order phase**: Asymptotic distance ≈ -15 (machine precision)
- **Chaos phase**: Asymptotic distance > -10
- **Transitions**: Sharp changes in asymptotic distance

## Generated Outputs

```
results/
├── training_history.json         # Loss and accuracy
├── chaos_analysis_results.pkl    # Complete results
├── analysis_summary.json         # Summary metrics
└── figures/
    ├── training_curves.png       # Basic training curves
    ├── chaos_dynamics.png        # Loss with asymptotic distances
    └── bifurcation_diagram.png   # Hidden state trajectories

checkpoints/
├── model_epoch_*.pt             # Checkpoints for each epoch
└── best_model.pt                # Best model
```

## Manual Inspection Guide

After running the experiment, inspect the generated plots for:

### 1. Multiple Descent Cycles
- Look at `training_curves.png`
- Check test loss in overfitting regime (after minimum)
- Count cycles of loss increase followed by sharp decrease

### 2. Order-Chaos Transitions
- Look at `chaos_dynamics.png`
- Identify regions where asymptotic distance is near -15 (order)
- Identify regions where asymptotic distance is above -10 (chaos)
- Note transitions between these regimes

### 3. Correlation Analysis
- Compare loss patterns with asymptotic distance changes
- Check if loss decreases coincide with transitions
- Note temporal relationships between dynamics

### 4. Bifurcation Patterns
- Look at `bifurcation_diagram.png`
- Order phases show convergence (tight clusters)
- Chaos phases show scattering (wide spread)

## Computational Requirements

**Training Time:**
- 1,000 epochs: ~2-6 hours (hardware dependent)

**Chaos Analysis:**  
- 200 epochs: ~4-8 hours

**Memory:**
- GPU: 4GB+ VRAM recommended
- RAM: 8GB+ system memory

**Tips:**
- Start with `--quick-test` for verification
- Use GPU if available
- Analyze fewer epochs initially

## Key Parameters

In `config.py`:

```python
# For faster experiments
MAX_EPOCHS = 100
NUM_TEST_SAMPLES = 100
END_EPOCH = 50

# For thorough analysis
MAX_EPOCHS = 1000
NUM_TEST_SAMPLES = 500
END_EPOCH = 200
```

## Troubleshooting

**CUDA out of memory**:
- Reduce `BATCH_SIZE` in config
- Use `DEVICE = 'cpu'`

**Dataset download fails**:
- Manually download from Kaggle
- Place in `data/` directory

**Long analysis time**:
- Reduce `NUM_TEST_SAMPLES`
- Analyze fewer epochs
- Use `--quick-test` mode

**Missing checkpoints**:
- Ensure training completed
- Check `checkpoints/` directory

## Analysis Checklist

- [ ] Training converges and shows overfitting
- [ ] Asymptotic distances calculated successfully
- [ ] Visualizations generated
- [ ] Manual inspection completed
- [ ] Patterns documented

## Notes

- This code provides the computational tools for chaos analysis
- Pattern detection requires manual inspection and judgment
- Results may vary based on random initialization
- Multiple runs may be needed for robust conclusions