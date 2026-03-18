"""
Main chaos analysis script - calculates asymptotic distances only
Manual inspection required for detecting patterns
"""

import os
import torch
import numpy as np
import json
from tqdm import tqdm

import config
from model import RNN
from config_saver import save_experiment_config
from data_loader import IMDBDataLoader
from asymptotic_analyzer import HiddenStateAnalyzer
from ftle_analyzer import FTLEBenettinAnalyzer
from seed_utils import HierarchicalSeedManager

import h5py

class AnalysisRunner:
    """Main analyzer class for calculating asymptotic distances"""
    
    def __init__(self, seed_manager=None):
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print("Analyzing chaos dynamics")
        self.seed_manager = seed_manager or HierarchicalSeedManager(config.RANDOM_SEED)
        
        # Initialize components
        self.data_loader = IMDBDataLoader(seed_manager=self.seed_manager)
        self.model = None
        self.hiddenstate_analyzer = None
        
        # Fixed analysis subset (prepared once for reproducibility & speed)
        self.sample_indices = None
        self.sample_tokens = None
        self.w0 = None
        self.ftle_analyzers = {}

        configured_window_lengths = getattr(config, 'FTLE_WINDOW_LENGTHS', [])
        self.ftle_window_lengths = sorted({int(w) for w in configured_window_lengths if int(w) > 0})
        if not self.ftle_window_lengths:
            raise ValueError("FTLE_WINDOW_LENGTHS must contain at least one positive integer.")
        configured_eps = getattr(config, 'FTLE_EPS', 1e-3)
        eps_list = configured_eps if isinstance(configured_eps, (list, tuple)) else [configured_eps]
        self.ftle_eps_values = [float(e) for e in eps_list if float(e) > 0]
        if not self.ftle_eps_values:
            raise ValueError("FTLE_EPS must contain at least one positive value.")
        # Results storage
        self.results = {
            'epochs': [],
            'train_loss': [],
            'test_loss': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'grad_norms': [],
            'analyzed_epochs': [],
            'bifurcation_data': [],
            'sample_indices': [],
            'ftle_eps_values': self.ftle_eps_values,
            'ftle_window_lengths': self.ftle_window_lengths,
            'ftle_mean_by_window': [],
        }

    def load_data_and_model(self):
        print("Loading data and RNN model...")
        
        # Load data
        X_train, X_test, y_train, y_test = self.data_loader.load_data()
        _, _, self.test_dataset = self.data_loader.create_data_loaders(X_train, X_test, y_train, y_test)
        
        self.model = RNN(
            vocab_size=self.data_loader.vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_classes=config.NUM_CLASSES,
            seed_manager=self.seed_manager,
        ).to(self.device)
        
        # Initialize chaos analyzer
        self.hiddenstate_analyzer = HiddenStateAnalyzer(
            self.model,
            self.device,
            seed_manager=self.seed_manager,
        )
        
        # Initialize FTLE analyzers (Benettin) for each (eps, window_length).
        for eps_val in self.ftle_eps_values:
            for window_length in self.ftle_window_lengths:
                key = (eps_val, window_length)
                self.ftle_analyzers[key] = FTLEBenettinAnalyzer(
                    self.model,
                    self.device,
                    eps=eps_val,
                    window_length=window_length,
                    zero_input_timesteps=config.ZERO_INPUT_TIMESTEPS,
                    burn_in=config.FTLE_BURN_IN,
                    seed_manager=self.seed_manager,
                )
        print(f"FTLE eps: {self.ftle_eps_values}, window lengths: {self.ftle_window_lengths}")
        self._prepare_analysis_samples()
        
        print(f"Data loaded. Test dataset size: {len(self.test_dataset)}")
        
    def _prepare_analysis_samples(self):
        sample_generator = self.seed_manager.torch_generator("analysis.sample_selection")
        w0_generator = self.seed_manager.torch_generator("analysis.ftle.w0")

        pad_token = 0
        
        # 先找出所有“完全没有 pad”的测试样本索引
        valid_indices = []
        for i in range(len(self.test_dataset)):
            tokens = self.test_dataset[i][0]
            if (tokens != pad_token).all().item():
                valid_indices.append(i)

        if len(valid_indices) == 0:
            raise ValueError("No test samples without padding were found.")
        else:
            print("valid_indices:" , len(valid_indices))
        
        # 从这些无 pad 样本里随机选 n 个
        n = min(int(config.NUM_TEST_SAMPLES), len(valid_indices))
        perm = torch.randperm(len(valid_indices), generator=sample_generator)[:n].tolist()
        indices = [valid_indices[j] for j in perm]

        # Token batch: [B, T] on analysis device.
        tokens = torch.stack([self.test_dataset[i][0] for i in indices], dim=0).to(self.device)

        # Fixed random direction per-sample.
        # LSTM: state is (h,c) -> 2H; GRU/RNN: state is h -> H.
        cell_type = getattr(self.model, "cell_type", "lstm").lower()
        state_dim = 2 * config.HIDDEN_SIZE if cell_type == "lstm" else config.HIDDEN_SIZE
        w0 = torch.randn(n, state_dim, generator=w0_generator).to(self.device)
        
        self.sample_indices = indices
        self.sample_tokens = tokens
        self.w0 = w0
        
    def load_training_history(self):
        """Load training history from file"""
        history_path = os.path.join(config.RESULTS_PATH, 'training_history.json')
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            self.results['epochs'] = history['epochs']
            self.results['train_loss'] = history['train_loss'] 
            self.results['test_loss'] = history['test_loss']
            self.results['train_accuracy'] = history['train_accuracy']
            self.results['test_accuracy'] = history['test_accuracy']
            
            if 'grad_norms' in history:
                self.results['grad_norms'] = history['grad_norms']
            
            print(f"Loaded training history for {len(self.results['epochs'])} epochs")
        else:
            print("No training history found.")
            return False
        
        return True
    
    def analyze_chaos_dynamics(self, start_epoch=1, end_epoch=None, interval=1, epochs_to_check=None):
        """
        Analyze chaos dynamics for selected epochs.

        Args:
            start_epoch: inclusive start epoch (1-indexed) used when epochs_to_check is None
            end_epoch: inclusive end epoch (1-indexed) used when epochs_to_check is None
            interval: stride used when epochs_to_check is None
            epochs_to_check: optional explicit list/tuple of epochs to analyze (1-indexed).
                When provided, it takes precedence over start/end/interval.
        """
        total_epochs = len(self.results.get('epochs', []))
        if total_epochs <= 0:
            raise ValueError("Training history is empty. Load training history before analysis.")

        if epochs_to_check is not None:
            if isinstance(epochs_to_check, (int, np.integer)):
                epochs_raw = [int(epochs_to_check)]
            else:
                epochs_raw = [int(e) for e in list(epochs_to_check)]
            # keep unique while preserving order
            seen = set()
            epochs_to_analyze = []
            for e in epochs_raw:
                if e in seen:
                    continue
                seen.add(e)
                if 1 <= e <= total_epochs:
                    epochs_to_analyze.append(e)
            if not epochs_to_analyze:
                raise ValueError(f"epochs_to_check has no valid epochs in [1, {total_epochs}].")
            print(f"Analyzing chaos dynamics for epochs: {epochs_to_analyze}")
        else:
            if end_epoch is None:
                end_epoch = total_epochs
            start_epoch = int(start_epoch)
            end_epoch = int(end_epoch)
            interval = int(interval)
            if interval <= 0:
                raise ValueError("interval must be a positive integer.")
            if start_epoch < 1:
                start_epoch = 1
            if end_epoch > total_epochs:
                end_epoch = total_epochs
            if start_epoch > end_epoch:
                raise ValueError(f"start_epoch ({start_epoch}) must be <= end_epoch ({end_epoch}).")

            print(f"Analyzing chaos dynamics from epoch {start_epoch} to {end_epoch} (interval={interval})...")
            epochs_to_analyze = list(range(start_epoch, end_epoch + 1, interval))
        
        bifurcation_data = []
        
        ftle_mean_by_window = []
        
        for epoch in tqdm(epochs_to_analyze, desc="Chaos analysis"):
            try:
                checkpoint_path = os.path.join(config.CHECKPOINT_PATH, f'model_epoch_{epoch}.pt')
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])

                epoch_ftle_means = []
                for eps_val in self.ftle_eps_values:
                    eps_row = []
                    for window_length in self.ftle_window_lengths:
                        _, ftle_avg = self.ftle_analyzers[(eps_val, window_length)].compute_ftle_benettin(
                            self.sample_tokens, w0=self.w0
                        )
                        eps_row.append(ftle_avg)
                    epoch_ftle_means.append(eps_row)
                ftle_mean_by_window.append(epoch_ftle_means)
                
                h_states_all_samples = self.hiddenstate_analyzer.calculate_hidden_states(
                    self.test_dataset, self.sample_indices, epoch
                )

                reduced_sums = self.hiddenstate_analyzer.calculate_reduced_sums_normalized(h_states_all_samples)
                bifurcation_data.append(reduced_sums)
                
            except FileNotFoundError:
                print(f"Checkpoint not found for epoch {epoch}")
                bifurcation_data.append([])
                nan_row = [[float("nan")] * len(self.ftle_window_lengths) for _ in self.ftle_eps_values]
                ftle_mean_by_window.append(nan_row)
                
        self.results['analyzed_epochs'] = epochs_to_analyze
        self.results['bifurcation_data'] = bifurcation_data
        self.results['sample_indices'] = self.sample_indices
        self.results['ftle_eps_values'] = self.ftle_eps_values
        self.results['ftle_window_lengths'] = self.ftle_window_lengths
        self.results['ftle_mean_by_window'] = ftle_mean_by_window
        
        print(f"Calculated asymptotic distances for {len(epochs_to_analyze)} epochs")
        print("Results saved for manual inspection")
        
        return epochs_to_analyze, bifurcation_data, ftle_mean_by_window
    
    def save_results(self):
        analyzed_epochs = self.results.get('analyzed_epochs', [])
        ftle_mean_by_window = np.array(self.results.get('ftle_mean_by_window', []), dtype=np.float64)
        finite_mask = np.isfinite(ftle_mean_by_window)  # works for 2D or 3D
        
        summary = {
            'summary': {
                'total_epochs': len(self.results['epochs']),
                'analyzed_epochs': len(analyzed_epochs),
                'min_test_loss': float(min(self.results['test_loss'])),
                'min_test_loss_epoch': int(np.argmin(self.results['test_loss']) + 1),
                'ftle_mean_min': float(np.min(ftle_mean_by_window[finite_mask])) if np.any(finite_mask) else float("nan"),
                'ftle_mean_max': float(np.max(ftle_mean_by_window[finite_mask])) if np.any(finite_mask) else float("nan"),
            },
            'training_curves': {
                'epochs': self.results['epochs'],
                'test_loss': self.results['test_loss'],
                'test_accuracy': self.results['test_accuracy']
            },
            'ftle': {
                'analyzed_epochs': analyzed_epochs,
                'eps_values': self.results.get('ftle_eps_values', []),
                'window_lengths': self.results.get('ftle_window_lengths', []),
                'ftle_mean_by_window': self.results.get('ftle_mean_by_window', []),
            },
        }
        
        summary_path = os.path.join(config.RESULTS_PATH, 'analysis_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_path}")

        save_experiment_config(extra={
            "run_type": "analysis",
            **summary.get("summary", {}),
            "ftle_eps_values": self.results.get("ftle_eps_values", []),
            "ftle_window_lengths": self.results.get("ftle_window_lengths", []),
        })
        
        # Save result as HDF5 for efficient storage
        # 1. 批量将列表转成 NumPy 数组（根据数据类型选合适的 dtype，进一步省空间）
        dtype_mapping = {
            'epochs': np.int16,
            'train_loss': np.float32,
            'test_loss': np.float32,
            'train_accuracy': np.float32,
            'test_accuracy': np.float32,
            'grad_norms': np.float32,
            'analyzed_epochs': np.int16,
            'bifurcation_data': np.float32,
            'sample_indices': np.int32,
            'ftle_eps_values': np.float64,
            'ftle_window_lengths': np.int16,
            'ftle_mean_by_window': np.float32,
        }
        
        # 列表 → NumPy 数组
        for key in self.results:
            self.results[key] = np.array(self.results[key], dtype=dtype_mapping.get(key, np.float32))
        
        # 2. 用 h5py 保存
        h5py_path = os.path.join(config.RESULTS_PATH, 'chaos_analysis_results.h5')
        with h5py.File(h5py_path, 'w') as f:
            for key, arr in tqdm(self.results.items(), total=len(self.results), desc="saved as HDF5"):
                f.create_dataset(
                    name=key,
                    data=arr,
                    compression='lzf',  # 使用 lzf 压缩，速度快且适合大多数场景
                    chunks=True  # 支持后续部分读取
                )
                
        print(f"Analysis Results saved to {h5py_path}")
