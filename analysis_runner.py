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
from model import LSTM
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
            'ftle_mean': [],
            'ftle_per_sample': []
        }

    def load_data_and_model(self):
        print("Loading data and RNN model...")
        
        # Load data
        X_train, X_test, y_train, y_test = self.data_loader.load_data()
        _, _, self.test_dataset = self.data_loader.create_data_loaders(X_train, X_test, y_train, y_test)
        
        self.model = LSTM(
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
        
        # Initialize FTLE analyzer (Benettin)
        self.ftle_analyzer = FTLEBenettinAnalyzer(
            self.model,
            self.device,
            eps=config.FTLE_EPS,
            window_length=config.FTLE_WINDOW_LENGTH,
            zero_input_timesteps=config.ZERO_INPUT_TIMESTEPS,
            burn_in=config.FTLE_BURN_IN,
            seed_manager=self.seed_manager,
        )
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

        # Fixed random direction per-sample (normalized to unit norm). 
        w0 = torch.randn(n, 2 * config.HIDDEN_SIZE, generator=w0_generator).to(self.device)
        w0 = w0 / w0.norm(dim=1, keepdim=True).clamp_min(1e-12)
        
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
    
    def analyze_chaos_dynamics(self, start_epoch=1, end_epoch=None, interval=1):   
        if end_epoch is None:
            end_epoch = len(self.results['epochs'])
            
        print(f"Analyzing chaos dynamics from epoch {start_epoch} to {end_epoch}...")
        
        epochs_to_analyze = list(range(start_epoch, min(end_epoch + 1, len(self.results['epochs']) + 1), interval))
        
        bifurcation_data = []
        
        ftle_mean = []
        ftle_per_sample = []
        
        for epoch in tqdm(epochs_to_analyze, desc="Chaos analysis"):
            try:
                checkpoint_path = os.path.join(config.CHECKPOINT_PATH, f'model_epoch_{epoch}.pt')
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])

                ftle_batch, ftle_avg = self.ftle_analyzer.compute_ftle_benettin(self.sample_tokens, w0=self.w0)
                ftle_per_sample.append(ftle_batch.detach().cpu().numpy())
                ftle_mean.append(ftle_avg)
                
                h_states_all_samples = self.hiddenstate_analyzer.calculate_hidden_states(
                    self.test_dataset, self.sample_indices, epoch
                )

                reduced_sums = self.hiddenstate_analyzer.calculate_reduced_sums_normalized(h_states_all_samples)
                bifurcation_data.append(reduced_sums)
                
            except FileNotFoundError:
                print(f"Checkpoint not found for epoch {epoch}")
                bifurcation_data.append([])
                
        self.results['analyzed_epochs'] = epochs_to_analyze
        self.results['bifurcation_data'] = bifurcation_data
        self.results['sample_indices'] = self.sample_indices
        self.results['ftle_mean'] = ftle_mean
        self.results['ftle_per_sample'] = ftle_per_sample
        
        print(f"Calculated asymptotic distances for {len(epochs_to_analyze)} epochs")
        print("Results saved for manual inspection")
        
        return epochs_to_analyze, bifurcation_data, ftle_mean, ftle_per_sample
    
    def save_results(self):
        analyzed_epochs = self.results.get('analyzed_epochs', [])
        ftle_mean = np.array(self.results.get('ftle_mean', []), dtype=np.float64)
        finite_mask = np.isfinite(ftle_mean)
        
        summary = {
            'summary': {
                'total_epochs': len(self.results['epochs']),
                'analyzed_epochs': len(analyzed_epochs),
                'min_test_loss': float(min(self.results['test_loss'])),
                'min_test_loss_epoch': int(np.argmin(self.results['test_loss']) + 1),
                'ftle_mean_min': float(np.min(ftle_mean[finite_mask])) if np.any(finite_mask) else float("nan"),
                'ftle_mean_max': float(np.max(ftle_mean[finite_mask])) if np.any(finite_mask) else float("nan"),
            },
            'training_curves': {
                'epochs': self.results['epochs'],
                'test_loss': self.results['test_loss'],
                'test_accuracy': self.results['test_accuracy']
            },
            'ftle': {
                'analyzed_epochs': analyzed_epochs,
                'ftle_mean': self.results.get('ftle_mean', []),
            },
        }
        
        summary_path = os.path.join(config.RESULTS_PATH, 'analysis_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_path}")
        
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
            'ftle_mean': np.float32,
            'ftle_per_sample': np.float32,
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
