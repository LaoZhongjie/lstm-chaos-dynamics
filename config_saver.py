"""
Save experiment configuration and model initialization info after each run.
"""

import os
import json
from datetime import datetime

import config


def _serialize_value(v):
    """Convert value to JSON-serializable form."""
    import numpy as np
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (np.integer, np.int64, np.int32)):
        return int(v)
    if isinstance(v, (np.floating, np.float64, np.float32)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_serialize_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _serialize_value(val) for k, val in v.items()}
    return str(v)


def collect_experiment_config(extra=None):
    """
    Collect all hyperparameters, config, and model initialization description.
    
    Returns a dict suitable for JSON serialization.
    """
    out = {
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {},
        "model_architecture": {},
        "model_init": {},
        "paths": {},
    }

    # Collect config attributes (skip private and callables)
    for key in dir(config):
        if key.startswith("_"):
            continue
        try:
            val = getattr(config, key)
            if callable(val):
                continue
            out["hyperparameters"][key] = _serialize_value(val)
        except Exception:
            pass

    # Model initialization (derived from config and code logic)
    cell_type = getattr(config, "RNN_CELL_TYPE", "lstm")
    out["model_architecture"] = {
        "cell_type": _serialize_value(cell_type),
        "module": f"torch.nn.{str(cell_type).upper()}",
        "notes": "Recurrent cell chosen via config.RNN_CELL_TYPE.",
    }
    out["model_init"] = {
        "embedding": (
            f"loaded from checkpoint ({config.PRETRAINED_CHECKPOINT}), requires_grad=False"
            if config.EMBEDDING_FIX
            else "uniform_(-0.1, 0.1), padding_idx=0 zeroed"
        ),
        "fc": (
            f"loaded from checkpoint ({config.PRETRAINED_CHECKPOINT}), requires_grad=False"
            if config.FC_FIX
            else "xavier_uniform_ (weight), zeros_ (bias)"
        ),
        "recurrent": {
            "cell_type": _serialize_value(cell_type),
            "weight_init": "PyTorch default (nn.<cell> parameters are left as constructed)",
            "bias_init": "PyTorch default",
        },
    }

    # Paths
    out["paths"] = {
        "DATA_PATH": config.DATA_PATH,
        "RESULTS_PATH": config.RESULTS_PATH,
        "CHECKPOINT_PATH": config.CHECKPOINT_PATH,
    }

    if extra:
        out["run_info"] = _serialize_value(extra)

    return out


def save_experiment_config(save_path=None, extra=None):
    """
    Save full experiment config and model init info to JSON.
    Call after training or analysis completes.

    Args:
        save_path: Output file path. Default: {RESULTS_PATH}/experiment_config.json
        extra: Optional dict of run-specific data (e.g., best_epoch, analyzed_epochs)
    """
    if save_path is None:
        save_path = os.path.join(config.RESULTS_PATH, "experiment_config.json")

    cfg = collect_experiment_config(extra)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"Experiment config saved to {save_path}")
