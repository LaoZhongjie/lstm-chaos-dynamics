"""
Utilities for hierarchical seed management.
"""

from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

import numpy as np
import torch


_UINT32_MOD = 2**32


def normalize_seed(seed: int) -> int:
    """Clamp any integer seed into [0, 2**32)."""
    return int(seed) % _UINT32_MOD


def derive_module_seed(global_seed: int, module_name: str, *, offset: int = 0) -> int:
    """Derive a stable module seed from global seed + module name (+ offset)."""
    if not module_name:
        raise ValueError("module_name must be a non-empty string")

    root = normalize_seed(global_seed)
    payload = f"{root}:{module_name}:{int(offset)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def set_global_seed(seed: int, *, deterministic: bool = True) -> int:
    """
    Set default RNG states for Python / NumPy / PyTorch.

    Returns:
        normalized seed in [0, 2**32).
    """
    seed = normalize_seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


@contextmanager
def isolated_seed(seed: int, *, deterministic: Optional[bool] = None) -> Iterator[int]:
    """
    Temporarily set all default RNG states to a seed, then fully restore states.
    """
    seed = normalize_seed(seed)

    py_state = random.getstate()
    np_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_available = torch.cuda.is_available()
    cuda_states = torch.cuda.get_rng_state_all() if cuda_available else None
    prev_cudnn_deterministic = torch.backends.cudnn.deterministic
    prev_cudnn_benchmark = torch.backends.cudnn.benchmark

    if deterministic is None:
        deterministic = prev_cudnn_deterministic

    set_global_seed(seed, deterministic=deterministic)
    try:
        yield seed
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if cuda_available and cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.backends.cudnn.deterministic = prev_cudnn_deterministic
        torch.backends.cudnn.benchmark = prev_cudnn_benchmark


@dataclass
class HierarchicalSeedManager:
    """
    Global -> module -> local seed helper.
    """

    global_seed: int
    deterministic: bool = True
    _module_seed_cache: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.global_seed = normalize_seed(self.global_seed)

    def apply_global_seed(self) -> int:
        return set_global_seed(self.global_seed, deterministic=self.deterministic)

    def module_seed(self, module_name: str, *, offset: int = 0) -> int:
        cache_key = f"{module_name}:{int(offset)}"
        if cache_key not in self._module_seed_cache:
            self._module_seed_cache[cache_key] = derive_module_seed(
                self.global_seed, module_name, offset=offset
            )
        return self._module_seed_cache[cache_key]

    def torch_generator(self, module_name: str, *, device="cpu", offset: int = 0) -> torch.Generator:
        device_type = torch.device(device).type
        generator = torch.Generator(device=device_type)
        generator.manual_seed(self.module_seed(module_name, offset=offset))
        return generator

    def numpy_rng(self, module_name: str, *, offset: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.module_seed(module_name, offset=offset))

    @contextmanager
    def local_seed(
        self, module_name: str, *, offset: int = 0, deterministic: Optional[bool] = None
    ) -> Iterator[int]:
        seed = self.module_seed(module_name, offset=offset)
        with isolated_seed(seed, deterministic=deterministic):
            yield seed
