from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

import config
from seed_utils import HierarchicalSeedManager


class FTLEBenettinAnalyzer:

    def __init__(
        self,
        model,
        device: torch.device,
        *,
        eps: float,
        window_length: int,
        zero_input_timesteps: int,
        burn_in: int = 0,
        clamp_min: float = 1e-30,
        seed_manager=None,
    ) -> None:
        if window_length <= 0:
            raise ValueError("window_length must be > 0")
        if zero_input_timesteps <= 0:
            raise ValueError("zero_input_timesteps must be > 0")
        if burn_in < 0:
            raise ValueError("burn_in must be >= 0")

        self.model = model
        self.device = device
        self.seed_manager = seed_manager or HierarchicalSeedManager(config.RANDOM_SEED)
        self._w0_generator = self.seed_manager.torch_generator("ftle.runtime.w0", device=device)
        self.window_length = int(window_length)
        self.zero_input_timesteps = int(zero_input_timesteps)
        self.burn_in = int(burn_in)
        self.clamp_min = float(clamp_min)

        # Keep eps as a device scalar to avoid host->device churn in tight loops.
        self.eps = torch.tensor(float(eps), dtype=torch.float32, device=device)

        # Model stores embedding_dim on the instance; keep a robust fallback.
        self.embedding_dim = int(getattr(model, "embedding_dim", config.EMBEDDING_DIM))

    def compute_ftle_benettin(
        self, sample_tokens: torch.Tensor, *, w0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Args:
            token_batch: LongTensor [B, T] on `self.device` (token indices).
            w0: Optional initial direction [B, 2H] in concatenated (h, c) space.
                For backward compatibility, [B, H] is also accepted (cell direction = 0).
                If provided, it will be normalized per-sample.

        Returns:
            ftle_per_sample: FloatTensor [B] (float32) on `self.device`.
            ftle_mean: Python float (mean over samples).
        """
        prev_mode = self.model.training
        self.model.eval()

        try:
            with torch.inference_mode():
                # 1) Natural-input segment: map tokens -> final LSTM state (h_T, c_T) (B, H)
                _, final_hc = self.model.get_hidden_output(sample_tokens)
                if not (isinstance(final_hc, tuple) and len(final_hc) == 2):
                    raise TypeError("Expected model.get_hidden_output() to return (output, (h, c)) for an LSTM model.")

                h_final, c_final = final_hc  # each: (1, B, H)
                h = h_final.squeeze(0)
                c = c_final.squeeze(0)
                
                batch_size, hidden_size = h.shape

                # 2) Initialize perturbation direction (unit-norm per sample)
                if w0 is None:
                    w0 = torch.randn(
                        batch_size,
                        2 * hidden_size,
                        generator=self._w0_generator,
                        device=self.device,
                        dtype=h.dtype,
                    )
                else:
                    w0 = w0.to(device=self.device, dtype=h.dtype)

                if w0.ndim != 2 or w0.shape[0] != batch_size:
                    raise ValueError(f"w0 must have shape [B, 2H] (or [B, H]); got {tuple(w0.shape)}.")

                if w0.shape[1] == hidden_size:
                    w0 = torch.cat([w0, torch.zeros_like(w0)], dim=1)
                elif w0.shape[1] != 2 * hidden_size:
                    raise ValueError(f"w0 must have shape [B, 2H] (or [B, H]); got {tuple(w0.shape)}.")

                w0 = w0 / w0.norm(dim=1, keepdim=True).clamp_min(1e-12)
                eps = self.eps.to(h.dtype)
                h_p = h + eps * w0[:, :hidden_size]
                c_p = c + eps * w0[:, hidden_size:]

                # 3) Block schedule for the zero-drive segment
                burn_blocks = math.ceil(self.burn_in / self.window_length) if self.burn_in > 0 else 0
                full_blocks, rem = divmod(self.zero_input_timesteps, self.window_length)
                block_lengths = [self.window_length] * full_blocks + ([rem] if rem else [])

                if burn_blocks >= len(block_lengths):
                    # Nothing left after burn-in; return zeros to make downstream handling easy.
                    ftle_zeros = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
                    return ftle_zeros, 0.0

                # Effective time (in steps) used for the exponent average
                t_eff = int(sum(block_lengths[burn_blocks:]))

                sum_logs = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
                eps64 = self.eps.to(torch.float64)

                # Pre-allocate zero-input blocks (2B because we evolve natural + perturbed together).
                zero_full = torch.zeros(
                    2 * batch_size,
                    self.window_length,
                    self.embedding_dim,
                    device=self.device,
                    dtype=h.dtype,
                )
                zero_rem = (
                    torch.zeros(
                        2 * batch_size,
                        rem,
                        self.embedding_dim,
                        device=self.device,
                        dtype=h.dtype,
                    )
                    if rem
                    else None
                )

                # 4) Benettin loop (block-wise evolve -> measure -> accumulate -> renormalize)
                for k, block_len in enumerate(block_lengths):
                    zero_block = zero_full if block_len == self.window_length else zero_rem
                    assert zero_block is not None

                    # Evolve both trajectories in one LSTM call (stacked batch: [base, perturbed]).
                    h_cat = torch.cat([h, h_p], dim=0).unsqueeze(0)  # (1, 2B, H)
                    c_cat = torch.cat([c, c_p], dim=0).unsqueeze(0)  # (1, 2B, H)
                    _, (hc_out, cc_out) = self.model.lstm(zero_block, (h_cat, c_cat))
                    hc_out = hc_out.squeeze(0)
                    cc_out = cc_out.squeeze(0)
                    h, h_p = hc_out[:batch_size], hc_out[batch_size:]
                    c, c_p = cc_out[:batch_size], cc_out[batch_size:]

                    # Separation and log-growth
                    delta = torch.cat([h_p - h, c_p - c], dim=1)  # (B, 2H)
                    r_t = delta.norm(dim=1).clamp_min(self.clamp_min)  # (B,)

                    if k >= burn_blocks:
                        sum_logs += torch.log(r_t.to(torch.float64) / eps64)

                    # Renormalize perturbed state along current separation direction.
                    dir_vec = delta / r_t.unsqueeze(1)
                    h_p = h + eps * dir_vec[:, :hidden_size]
                    c_p = c + eps * dir_vec[:, hidden_size:]

                ftle_per_sample = (sum_logs / float(t_eff)).to(torch.float32)
                ftle_mean = float(ftle_per_sample.mean().item())
                return ftle_per_sample, ftle_mean
        finally:
            if prev_mode:
                self.model.train()
