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
        self.noise_generator = self.seed_manager.torch_generator("analysis.asymptotic.noise", device=device)

    def calculate_hidden_states(self, test_dataset, indices, epoch, zero_input_timesteps=config.ZERO_INPUT_TIMESTEPS):
        hidden_states_all_samples = []
        hidden_states_perturbed_all_samples = []
        
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
                final_hidden_state, final_cell_state = final_hc
                
                # Create perturbed initial hidden state
                noise = torch.randn(
                    final_hidden_state.shape,
                    generator=self.noise_generator,
                    device=final_hidden_state.device,
                    dtype=final_hidden_state.dtype,
                ) * config.NOISE_SCALE
                final_hidden_state_perturbed = final_hidden_state + noise
                final_hc_perturbed = (final_hidden_state_perturbed, final_cell_state)
                
                # Continue iteration with zero inputs
                hidden_states = self.model.continue_iteration(
                    final_hc, zero_input_timesteps, config.EMBEDDING_DIM
                )                     
                
                hidden_states_perturbed = self.model.continue_iteration(
                    final_hc_perturbed, zero_input_timesteps, config.EMBEDDING_DIM
                )
                
                hidden_states = hidden_states.squeeze(0)
                hidden_states_perturbed = hidden_states_perturbed.squeeze(0)
                
                hidden_states_all_samples.append(hidden_states)
                hidden_states_perturbed_all_samples.append(hidden_states_perturbed)
        
        
        return hidden_states_all_samples, hidden_states_perturbed_all_samples

    def calculate_perturbed_distance(self, hidden_states_all_samples, hidden_states_perturbed_all_samples):
        """Calculate log-scale trajectories from hidden states."""
        perturbed_distance_all_samples = []
        for h_states, h_states_perturbed in zip(hidden_states_all_samples, hidden_states_perturbed_all_samples):
            # Calculate distances for each timestep
            perturbed_distance = torch.norm(h_states_perturbed - h_states, p=2, dim=1).cpu().numpy() #shape(zero_input_timesteps)
            perturbed_distance_adjusted = perturbed_distance + np.exp(config.MACHINE_PRECISION_THRESHOLD)
            perturbed_distance_all_samples.append(perturbed_distance_adjusted)   #shape(num_samples, zero_input_timesteps)
        
        return np.log(np.array(perturbed_distance_all_samples, dtype=np.float32))

    def calculate_mean_final_perturbed_distance(self, perturbed_distance_all_samples):
        """Calculate mean asymptotic distance for final timestep."""
        return np.mean(perturbed_distance_all_samples[:, -1])  #scalar

    def calculate_reduced_sums_normalized(self, hidden_states_all_samples):
        """Calculate normalized reduced sums from hidden states."""
        reduced_sums = []
        for h_states in hidden_states_all_samples:
            reduced_sum = torch.sum(h_states, dim=1).detach().cpu().numpy()
            reduced_sums.append(reduced_sum)
        
        reduced_sums = np.array(reduced_sums, dtype=np.float32)  #shape(zero_input_timesteps)
        return reduced_sums - np.mean(reduced_sums, axis=0)

    def analyze_asymptotic_distance(self, hidden_states_all_samples, hidden_states_perturbed_all_samples):
        
        # Calculate formatted trajectories
        perturbed_distance_all_samples = self.calculate_perturbed_distance(
            hidden_states_all_samples, hidden_states_perturbed_all_samples
        )

        # Calculate asymptotic distance
        mean_final_perturbed_distance = self.calculate_mean_final_perturbed_distance(perturbed_distance_all_samples)

        # Calculate normalized reduced sums
        reduced_sums_normalized = self.calculate_reduced_sums_normalized(hidden_states_all_samples)

        return perturbed_distance_all_samples, mean_final_perturbed_distance, reduced_sums_normalized
