import os
import torch
import numpy as np
from tqdm import tqdm
import config
import numbers
from seed_utils import HierarchicalSeedManager

class HiddenStateAnalyzer:
    def __init__(self, model, device, seed_manager=None):
        self.model = model
        self.device = device
        self.seed_manager = seed_manager or HierarchicalSeedManager(config.RANDOM_SEED)

    def calculate_hidden_states(self, test_dataset, indices, epoch, zero_input_timesteps=config.ZERO_INPUT_TIMESTEPS):
        hidden_states_all_samples = []
        
        with torch.no_grad():
            for idx in tqdm(indices, desc="Processing hidden states", leave=False):
                # Accept either Python ints or scalar tensors.
                if isinstance(idx, torch.Tensor):
                    idx = int(idx.item())
                elif isinstance(idx, numbers.Integral) or isinstance(idx, np.integer):
                    idx = int(idx)
                else:
                    raise TypeError(f"Index must be an int or scalar tensor; got {type(idx)!r}")
                
                # Get sample
                item = test_dataset[idx]
                # IMDBDataset returns (tokens, label, length); other datasets may return (tokens, label).
                if isinstance(item, (tuple, list)):
                    sample = item[0]
                else:
                    sample = item
                sample = sample.unsqueeze(0).to(self.device)
                
                # Process input
                _, final_hc = self.model.get_hidden_output(sample)
                
                # Continue iteration with zero inputs
                hidden_states = self.model.continue_iteration(
                    final_hc, zero_input_timesteps, config.EMBEDDING_DIM
                )                     
                hidden_states = hidden_states.squeeze(0)
                
                hidden_states_all_samples.append(hidden_states)
        
        return hidden_states_all_samples

    def calculate_reduced_sums_normalized(self, hidden_states_all_samples):
        """Calculate normalized reduced sums from hidden states."""
        reduced_sums = []
        for h_states in hidden_states_all_samples:
            reduced_sum = torch.sum(h_states, dim=1).detach().cpu().numpy()
            reduced_sums.append(reduced_sum)
        
        reduced_sums = np.array(reduced_sums, dtype=np.float32)  #shape(zero_input_timesteps)
        return reduced_sums - np.mean(reduced_sums, axis=0)
