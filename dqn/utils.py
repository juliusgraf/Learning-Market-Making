# rl_auctions/dqn/utils.py
from __future__ import annotations

from typing import Any, Optional
import numpy as np
import torch

__all__ = ["to_tensor", "mask_logits"]


def to_tensor(x: Any, device: str):
    """
    Convert input to a torch.Tensor on `device` with a sensible dtype:
      - integers -> int64
      - bools    -> bool
      - floats   -> float32 (default)

    Supports torch.Tensor, numpy arrays, lists, and scalars.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)

    if isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.integer):
            t = torch.from_numpy(x.astype(np.int64, copy=False))
            return t.to(device)
        if np.issubdtype(x.dtype, np.bool_):
            t = torch.from_numpy(x.astype(np.bool_, copy=False))
            return t.to(device)
        # float / other -> float32
        t = torch.from_numpy(x.astype(np.float32, copy=False))
        return t.to(device)

    # Python scalar or list/tuple
    if isinstance(x, (list, tuple)):
        # Heuristic: if any float -> float32, else int64, else bool
        if any(isinstance(v, float) for v in x):
            return torch.tensor(x, dtype=torch.float32, device=device)
        if any(isinstance(v, bool) for v in x):
            return torch.tensor(x, dtype=torch.bool, device=device)
        return torch.tensor(x, dtype=torch.int64, device=device)

    if isinstance(x, bool):
        return torch.tensor(x, dtype=torch.bool, device=device)
    if isinstance(x, int):
        return torch.tensor(x, dtype=torch.int64, device=device)
    # default float32
    return torch.tensor(x, dtype=torch.float32, device=device)


def mask_logits(logits: torch.Tensor, mask: Optional[torch.Tensor], fill: float = -1e9) -> torch.Tensor:
    """
    Apply an action mask to logits/Q-values.

    Args:
        logits: Tensor of shape (B, A) or (A,).
        mask:   Boolean tensor with True for ALLOWED actions.
                Shapes supported: (A,) or (B, A). Will broadcast when possible.
                If None, logits are returned unchanged.
        fill:   Value to place where mask is False (e.g., a large negative).

    Returns:
        Tensor with same shape as logits where invalid actions are set to `fill`.
    """
    if mask is None:
        return logits

    # Ensure boolean mask on the same device
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, device=logits.device)
    else:
        mask = mask.to(device=logits.device)

    if mask.dtype != torch.bool:
        # Interpret non-bool masks: >0 => True
        mask = mask > 0

    # If logits is (A,) and mask is (A,) → fine.
    # If logits is (B, A) and mask is (A,) → expand.
    if logits.dim() == 2 and mask.dim() == 1 and mask.shape[-1] == logits.shape[-1]:
        mask = mask.unsqueeze(0).expand_as(logits)
    # Otherwise rely on broadcasting; torch.where will broadcast if possible.

    # Use where to avoid in-place on shared tensors
    return torch.where(mask, logits, torch.full_like(logits, fill))
