from __future__ import annotations

import math
from typing import Optional, Tuple, Union

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
        self._autonomous_init_generator = self.seed_manager.torch_generator("ftle.autonomous.init", device=device)
        self.window_length = int(window_length)
        self.zero_input_timesteps = int(zero_input_timesteps)
        self.burn_in = int(burn_in)
        self.clamp_min = float(clamp_min)
        self.cell_type = getattr(model, "cell_type", "lstm").lower()
        
        # Keep eps in float64 for FTLE arithmetic (logs/norm ratios).
        # Note: the RNN forward must remain float32/float16; we cast eps to state dtype when perturbing.
        self.eps = torch.tensor(float(eps), dtype=torch.float64, device=device)

        # Model stores embedding_dim on the instance; keep a robust fallback.
        self.embedding_dim = int(getattr(model, "embedding_dim", config.EMBEDDING_DIM))

    def _unpack_initial_hc_to_batch(
        self,
        initial_hc: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_size: int,
        hidden_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (h, c) each [B, H] on the same device/dtype as inputs; c is zeros for GRU/RNN."""
        if self.cell_type == "lstm":
            h0, c0 = initial_hc
            h = h0.squeeze(0)
            c = c0.squeeze(0)
        else:
            h = initial_hc.squeeze(0)
            c = torch.zeros_like(h)
        if h.shape != (batch_size, hidden_size) or c.shape != (batch_size, hidden_size):
            raise ValueError(
                f"initial_hc must match batch_size={batch_size}, hidden_size={hidden_size}; "
                f"got h.shape={tuple(h.shape)}, c.shape={tuple(c.shape)}."
            )
        return h, c

    def compute_ftle_benettin(
        self,
        sample_tokens: torch.Tensor,
        *,
        w0: Optional[torch.Tensor] = None,
        initial_hc: Optional[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Args:
            token_batch: LongTensor [B, T] on `self.device` (token indices).
            w0: Optional initial direction [B, 2H] in concatenated (h, c) space.
                For backward compatibility, [B, H] is also accepted (cell direction = 0).
                If provided, it will be normalized per-sample.
            initial_hc: Optional RNN state before the token sequence: LSTM (h0, c0) each (1, B, H);
                GRU/RNN h0 (1, B, H). If None, zeros (same as model.init_hidden).

        Returns:
            ftle_per_sample: FloatTensor [B] (float32) on `self.device` (computed in float64, returned as float32).
            ftle_mean: Python float (mean over samples).
        """
        prev_mode = self.model.training
        self.model.eval()

        try:
            with torch.inference_mode():
                # 1) Natural-input segment: map tokens -> final recurrent state
                # LSTM: final_state = (h_T, c_T); GRU/RNN: final_state = h_T
                _, final_state = self.model.get_hidden_output(sample_tokens, hidden=initial_hc)

                if isinstance(final_state, tuple) and len(final_state) == 2:
                    h_final, c_final = final_state  # (1,B,H) each
                    h = h_final.squeeze(0)
                    c = c_final.squeeze(0)
                else:
                    h = final_state.squeeze(0)
                    c = torch.zeros_like(h)
                
                batch_size, hidden_size = h.shape

                # 2) Initialize perturbation direction (unit-norm per sample)
                state_dim = 2 * hidden_size if self.cell_type == "lstm" else hidden_size
                if w0 is None:
                    w0 = torch.randn(
                        batch_size,
                        state_dim,
                        generator=self._w0_generator,
                        device=self.device,
                        dtype=torch.float64,
                    )
                else:
                    w0 = w0.to(device=self.device, dtype=torch.float64)

                if w0.ndim != 2 or w0.shape[0] != batch_size:
                    raise ValueError(f"w0 must have shape [B, {state_dim}] (or [B, H]/[B,2H]); got {tuple(w0.shape)}.")

                if self.cell_type == "lstm":
                    if w0.shape[1] == hidden_size:
                        w0 = torch.cat([w0, torch.zeros_like(w0)], dim=1)
                    elif w0.shape[1] != 2 * hidden_size:
                        raise ValueError(f"w0 must have shape [B, 2H] (or [B, H]); got {tuple(w0.shape)}.")
                else:
                    if w0.shape[1] != hidden_size:
                        raise ValueError(f"w0 must have shape [B, H] for GRU/RNN; got {tuple(w0.shape)}.")

                # Normalize in float64, then cast direction to state dtype for evolution.
                w0 = w0 / w0.norm(dim=1, keepdim=True).clamp_min(1e-12)
                eps_state = self.eps.to(dtype=h.dtype)
                w0_state = w0.to(dtype=h.dtype)
                # Initial perturbed state along w0
                if self.cell_type == "lstm":
                    h_p = h + eps_state * w0_state[:, :hidden_size]
                    c_p = c + eps_state * w0_state[:, hidden_size:]
                else:
                    # GRU / RNN: only perturb hidden state
                    h_p = h + eps_state * w0_state
                    c_p = c

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
                eps64 = self.eps  # float64 device scalar

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

                    # Evolve both trajectories in one recurrent call (stacked batch: [base, perturbed]).
                    h_cat = torch.cat([h, h_p], dim=0).unsqueeze(0)  # (1, 2B, H)
                    if self.cell_type == "lstm":
                        c_cat = torch.cat([c, c_p], dim=0).unsqueeze(0)  # (1, 2B, H)
                        _, (hc_out, cc_out) = self.model.rnn(zero_block, (h_cat, c_cat))
                        hc_out = hc_out.squeeze(0)
                        cc_out = cc_out.squeeze(0)
                        h, h_p = hc_out[:batch_size], hc_out[batch_size:]
                        c, c_p = cc_out[:batch_size], cc_out[batch_size:]
                    else:
                        _, h_out = self.model.rnn(zero_block, h_cat)
                        h_out = h_out.squeeze(0)
                        h, h_p = h_out[:batch_size], h_out[batch_size:]
                        c = torch.zeros_like(h)
                        c_p = torch.zeros_like(h_p)

                    # Separation and log-growth
                    # Compute FTLE arithmetic in float64 (but keep states in float32 for RNN).
                    dh = (h_p - h).to(torch.float64)
                    if self.cell_type == "lstm":
                        dc = (c_p - c).to(torch.float64)
                        delta64 = torch.cat([dh, dc], dim=1)  # (B, 2H)
                    else:
                        delta64 = dh  # (B, H)
                    r_t64 = delta64.norm(dim=1).clamp_min(self.clamp_min)  # (B,) float64

                    if k >= burn_blocks:
                        sum_logs += torch.log(r_t64 / eps64)

                    # Renormalize perturbed state along current separation direction.
                    dir_vec64 = delta64 / r_t64.unsqueeze(1)  # float64 unit direction
                    dir_vec_state = dir_vec64.to(dtype=h.dtype)
                    if self.cell_type == "lstm":
                        h_p = h + eps_state * dir_vec_state[:, :hidden_size]
                        c_p = c + eps_state * dir_vec_state[:, hidden_size:]
                    else:
                        # GRU / RNN: only hidden state is renormalized
                        h_p = h + eps_state * dir_vec_state
                        c_p = c

                ftle_per_sample = (sum_logs / float(t_eff)).to(torch.float32)
                ftle_mean = float(ftle_per_sample.mean().item())
                return ftle_per_sample, ftle_mean
        finally:
            if prev_mode:
                self.model.train()

    def compute_ftle_autonomous(
        self,
        batch_size: int,
        *,
        w0: Optional[torch.Tensor] = None,
        init_scale: float = 0.1,
        initial_hc: Optional[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Autonomous FTLE: same zero-input Benettin flow, but with random initial state
        (no token sequence). Measures intrinsic chaos of the autonomous dynamics
        f(h, c; x=0) from random starting points.

        Args:
            batch_size: number of independent trajectories.
            w0: optional initial direction [B, 2H] or [B, H]. If None, random per sample.
            init_scale: scale for random initial (h, c) when initial_hc is None. Default 0.1.
            initial_hc: If set, use this (h0, c0) instead of drawing a new random state (same layout as model.init_hidden).

        Returns:
            ftle_per_sample: [B] on device. ftle_mean: float.
        """
        prev_mode = self.model.training
        self.model.eval()

        try:
            with torch.inference_mode():
                hidden_size = getattr(self.model, "hidden_size", config.HIDDEN_SIZE)

                # 1) Initial state: shared initial_hc or legacy random draw
                if initial_hc is not None:
                    h, c = self._unpack_initial_hc_to_batch(initial_hc, batch_size, hidden_size)
                else:
                    h = init_scale * torch.randn(
                        batch_size, hidden_size,
                        generator=self._autonomous_init_generator,
                        device=self.device,
                        dtype=torch.float32,
                    )
                    if self.cell_type == "lstm":
                        c = init_scale * torch.randn(
                            batch_size, hidden_size,
                            generator=self._autonomous_init_generator,
                            device=self.device,
                            dtype=torch.float32,
                        )
                    else:
                        c = torch.zeros_like(h)

                # 2) Same w0/perturbation setup as 0-input
                state_dim = 2 * hidden_size if self.cell_type == "lstm" else hidden_size
                if w0 is None:
                    w0 = torch.randn(
                        batch_size, state_dim,
                        generator=self._w0_generator,
                        device=self.device,
                        dtype=torch.float64,
                    )
                else:
                    w0 = w0.to(device=self.device, dtype=torch.float64)

                if w0.ndim != 2 or w0.shape[0] != batch_size:
                    raise ValueError(f"w0 must have shape [B, {state_dim}]; got {tuple(w0.shape)}.")

                if self.cell_type == "lstm":
                    if w0.shape[1] == hidden_size:
                        w0 = torch.cat([w0, torch.zeros_like(w0)], dim=1)
                    elif w0.shape[1] != 2 * hidden_size:
                        raise ValueError(f"w0 must have shape [B, 2H] or [B, H]; got {tuple(w0.shape)}.")
                else:
                    if w0.shape[1] != hidden_size:
                        raise ValueError(f"w0 must have shape [B, H]; got {tuple(w0.shape)}.")

                w0 = w0 / w0.norm(dim=1, keepdim=True).clamp_min(1e-12)
                eps_state = self.eps.to(dtype=h.dtype)
                w0_state = w0.to(dtype=h.dtype)
                if self.cell_type == "lstm":
                    h_p = h + eps_state * w0_state[:, :hidden_size]
                    c_p = c + eps_state * w0_state[:, hidden_size:]
                else:
                    h_p = h + eps_state * w0_state
                    c_p = c

                # 3) Zero-input block schedule (same as 0-input FTLE)
                burn_blocks = math.ceil(self.burn_in / self.window_length) if self.burn_in > 0 else 0
                full_blocks, rem = divmod(self.zero_input_timesteps, self.window_length)
                block_lengths = [self.window_length] * full_blocks + ([rem] if rem else [])

                if burn_blocks >= len(block_lengths):
                    ftle_zeros = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
                    return ftle_zeros, 0.0

                t_eff = int(sum(block_lengths[burn_blocks:]))
                sum_logs = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
                eps64 = self.eps

                zero_full = torch.zeros(
                    2 * batch_size, self.window_length, self.embedding_dim,
                    device=self.device, dtype=h.dtype,
                )
                zero_rem = (
                    torch.zeros(2 * batch_size, rem, self.embedding_dim, device=self.device, dtype=h.dtype)
                    if rem else None
                )

                # 4) Benettin loop (zero input)
                for k, block_len in enumerate(block_lengths):
                    zero_block = zero_full if block_len == self.window_length else zero_rem
                    assert zero_block is not None

                    h_cat = torch.cat([h, h_p], dim=0).unsqueeze(0)
                    if self.cell_type == "lstm":
                        c_cat = torch.cat([c, c_p], dim=0).unsqueeze(0)
                        _, (hc_out, cc_out) = self.model.rnn(zero_block, (h_cat, c_cat))
                        hc_out, cc_out = hc_out.squeeze(0), cc_out.squeeze(0)
                        h, h_p = hc_out[:batch_size], hc_out[batch_size:]
                        c, c_p = cc_out[:batch_size], cc_out[batch_size:]
                    else:
                        _, h_out = self.model.rnn(zero_block, h_cat)
                        h_out = h_out.squeeze(0)
                        h, h_p = h_out[:batch_size], h_out[batch_size:]
                        c, c_p = torch.zeros_like(h), torch.zeros_like(h_p)

                    dh = (h_p - h).to(torch.float64)
                    if self.cell_type == "lstm":
                        dc = (c_p - c).to(torch.float64)
                        delta64 = torch.cat([dh, dc], dim=1)
                    else:
                        delta64 = dh
                    r_t64 = delta64.norm(dim=1).clamp_min(self.clamp_min)

                    if k >= burn_blocks:
                        sum_logs += torch.log(r_t64 / eps64)

                    dir_vec64 = delta64 / r_t64.unsqueeze(1)
                    dir_vec_state = dir_vec64.to(dtype=h.dtype)
                    if self.cell_type == "lstm":
                        h_p = h + eps_state * dir_vec_state[:, :hidden_size]
                        c_p = c + eps_state * dir_vec_state[:, hidden_size:]
                    else:
                        h_p = h + eps_state * dir_vec_state
                        c_p = c

                ftle_per_sample = (sum_logs / float(t_eff)).to(torch.float32)
                ftle_mean = float(ftle_per_sample.mean().item())
                return ftle_per_sample, ftle_mean
        finally:
            if prev_mode:
                self.model.train()

    def compute_ftle_driven(
        self,
        sample_tokens: torch.Tensor,
        *,
        w0: Optional[torch.Tensor] = None,
        repeat_factor: Optional[int] = None,
        initial_hc: Optional[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Driven FTLE: Benettin along a trajectory driven by the token sequence.
        The sequence is cycled (repeat_factor x) to reach ~zero_input_timesteps length.
        Both reference and perturbed trajectories receive the SAME tokens at each step.
        Initial state is zero by default, or initial_hc when provided (match 0-input / autonomous).

        Args:
            sample_tokens: LongTensor [B, T] token indices.
            w0: optional direction [B, 2H] or [B, H]. If None, random per sample.
            repeat_factor: times to repeat the sequence. If None, use ceil(zero_input_timesteps / T).
            initial_hc: Optional (h0, c0) for LSTM or h0 for GRU/RNN, each (1, B, H). If None, zeros.

        Returns:
            ftle_per_sample: [B]. ftle_mean: float.
        """
        prev_mode = self.model.training
        self.model.eval()

        try:
            with torch.inference_mode():
                batch_size, seq_len = sample_tokens.shape
                hidden_size = getattr(self.model, "hidden_size", config.HIDDEN_SIZE)

                # 1) Cycle tokens to reach zero_input_timesteps length
                target_steps = self.zero_input_timesteps
                if repeat_factor is None:
                    repeat_factor = max(1, math.ceil(target_steps / seq_len))
                n_repeat = repeat_factor
                tokens_cycled = sample_tokens.repeat(1, n_repeat)  # [B, seq_len * n_repeat]
                total_len = tokens_cycled.shape[1]
                # Clip to target_steps for consistent comparison with 0-input
                if total_len > target_steps:
                    tokens_cycled = tokens_cycled[:, :target_steps]
                total_len = tokens_cycled.shape[1]

                # 2) Initial state (zeros or shared initial_hc)
                if initial_hc is not None:
                    h, c = self._unpack_initial_hc_to_batch(initial_hc, batch_size, hidden_size)
                else:
                    h = torch.zeros(batch_size, hidden_size, device=self.device, dtype=torch.float32)
                    c = torch.zeros(batch_size, hidden_size, device=self.device, dtype=torch.float32)

                # 3) w0 setup
                state_dim = 2 * hidden_size if self.cell_type == "lstm" else hidden_size
                if w0 is None:
                    w0 = torch.randn(
                        batch_size, state_dim,
                        generator=self._w0_generator,
                        device=self.device,
                        dtype=torch.float64,
                    )
                else:
                    w0 = w0.to(device=self.device, dtype=torch.float64)

                if w0.ndim != 2 or w0.shape[0] != batch_size:
                    raise ValueError(f"w0 must have shape [B, {state_dim}]; got {tuple(w0.shape)}.")

                if self.cell_type == "lstm":
                    if w0.shape[1] == hidden_size:
                        w0 = torch.cat([w0, torch.zeros_like(w0)], dim=1)
                    elif w0.shape[1] != 2 * hidden_size:
                        raise ValueError(f"w0 must have shape [B, 2H] or [B, H]; got {tuple(w0.shape)}.")
                else:
                    if w0.shape[1] != hidden_size:
                        raise ValueError(f"w0 must have shape [B, H]; got {tuple(w0.shape)}.")

                w0 = w0 / w0.norm(dim=1, keepdim=True).clamp_min(1e-12)
                eps_state = self.eps.to(dtype=h.dtype)
                w0_state = w0.to(dtype=h.dtype)
                if self.cell_type == "lstm":
                    h_p = h + eps_state * w0_state[:, :hidden_size]
                    c_p = c + eps_state * w0_state[:, hidden_size:]
                else:
                    h_p = h + eps_state * w0_state
                    c_p = c

                # 4) Block schedule over driven sequence
                burn_blocks = math.ceil(self.burn_in / self.window_length) if self.burn_in > 0 else 0
                full_blocks, rem = divmod(total_len, self.window_length)
                block_lengths = [self.window_length] * full_blocks + ([rem] if rem else [])

                if burn_blocks >= len(block_lengths):
                    ftle_zeros = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
                    return ftle_zeros, 0.0

                t_eff = int(sum(block_lengths[burn_blocks:]))
                sum_logs = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
                eps64 = self.eps

                # 5) Benettin loop with driven input
                pos = 0
                for k, block_len in enumerate(block_lengths):
                    token_slice = tokens_cycled[:, pos : pos + block_len]  # [B, block_len]
                    pos += block_len
                    embedded = self.model.embedding(token_slice)  # [B, block_len, E]
                    # Same input for base and perturbed: [2B, block_len, E]
                    driven_block = embedded.repeat(2, 1, 1)

                    h_cat = torch.cat([h, h_p], dim=0).unsqueeze(0)
                    if self.cell_type == "lstm":
                        c_cat = torch.cat([c, c_p], dim=0).unsqueeze(0)
                        _, (hc_out, cc_out) = self.model.rnn(driven_block, (h_cat, c_cat))
                        hc_out, cc_out = hc_out.squeeze(0), cc_out.squeeze(0)
                        h, h_p = hc_out[:batch_size], hc_out[batch_size:]
                        c, c_p = cc_out[:batch_size], cc_out[batch_size:]
                    else:
                        _, h_out = self.model.rnn(driven_block, h_cat)
                        h_out = h_out.squeeze(0)
                        h, h_p = h_out[:batch_size], h_out[batch_size:]
                        c, c_p = torch.zeros_like(h), torch.zeros_like(h_p)

                    dh = (h_p - h).to(torch.float64)
                    if self.cell_type == "lstm":
                        dc = (c_p - c).to(torch.float64)
                        delta64 = torch.cat([dh, dc], dim=1)
                    else:
                        delta64 = dh
                    r_t64 = delta64.norm(dim=1).clamp_min(self.clamp_min)

                    if k >= burn_blocks:
                        sum_logs += torch.log(r_t64 / eps64)

                    dir_vec64 = delta64 / r_t64.unsqueeze(1)
                    dir_vec_state = dir_vec64.to(dtype=h.dtype)
                    if self.cell_type == "lstm":
                        h_p = h + eps_state * dir_vec_state[:, :hidden_size]
                        c_p = c + eps_state * dir_vec_state[:, hidden_size:]
                    else:
                        h_p = h + eps_state * dir_vec_state
                        c_p = c

                ftle_per_sample = (sum_logs / float(t_eff)).to(torch.float32)
                ftle_mean = float(ftle_per_sample.mean().item())
                return ftle_per_sample, ftle_mean
        finally:
            if prev_mode:
                self.model.train()
