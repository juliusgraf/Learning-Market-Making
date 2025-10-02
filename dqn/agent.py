from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterable, Union, Sequence
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import DQN


@dataclass
class EpsilonScheduler:
    """Linear ε schedule from start → end over decay_steps."""
    start: float
    end: float
    decay_steps: int

    def value(self, step: int) -> float:
        if self.decay_steps <= 0:
            return float(self.end)
        step = max(0, int(step))
        frac = min(1.0, step / float(self.decay_steps))
        eps = self.start + (self.end - self.start) * frac
        return float(max(0.0, min(1.0, eps)))


class DQNAgent:
    """
    Epsilon-greedy DQN with Polyak target updates.

    act(obs, step, eps_sched, admissible=None) -> int
      - obs: np.ndarray (obs_dim,)
      - admissible: None, or
          * boolean mask of shape (n_actions,), True = allowed
          * iterable of allowed indices

    learn(batch, gamma) -> dict(loss=..., q_mean=..., target_mean=...)
      - batch from ReplayBuffer.sample
    """

    def __init__(
        self,
        q_net: DQN,
        target_net: DQN,
        n_actions: int,
        lr: float,
        tau_target: float,
        device: str = "cpu",
        grad_clip_norm: Optional[float] = 10.0,
    ):
        self.q = q_net.to(device)
        self.target = target_net.to(device)
        self.n_actions = int(n_actions)
        self.device = device
        self.tau = float(tau_target)
        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.grad_clip_norm = grad_clip_norm

        # target ← q (hard copy at init)
        self._hard_update()

        # evaluation mode for target net
        self.target.eval()

        # Random source for ε exploration
        self._rng = random.Random(1234)

    # -------------------- Acting --------------------

    def act(
        self,
        obs: np.ndarray,
        step: int,
        eps_sched: EpsilonScheduler,
        admissible: Optional[Union[np.ndarray, Sequence[int]]] = None,
    ) -> int:
        """ε-greedy action with optional mask over admissible actions."""
        eps = float(eps_sched.value(step))

        # Random action path
        if self._rng.random() < eps:
            if admissible is None:
                return int(self._rng.randrange(self.n_actions))
            mask_idx = self._allowed_indices_from_admissible(admissible)
            if len(mask_idx) == 0:
                return int(self._rng.randrange(self.n_actions))
            return int(self._rng.choice(mask_idx))

        # Greedy path
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)                          # (1, obs_dim)
            q = self.q(x)                                    # (1, A)
            q = q.squeeze(0)                                 # (A,)

            if admissible is not None:
                mask = self._mask_from_admissible(admissible)   # (A,)
                # set invalid actions to very negative
                invalid = ~mask
                if invalid.any():
                    q = q.clone()
                    q[invalid] = -1e9

            a = int(torch.argmax(q).item())
            return a

    # -------------------- Learning --------------------

    def learn(self, batch: Dict[str, np.ndarray], gamma: float) -> Dict[str, float]:
        """
        One TD step with Huber loss:
            y = r + γ * (1 - done) * max_a' Q_target(s', a')
            L = Huber( Q(s,a), y )
        """
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)          # (B, obs_dim)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)    # (B,)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)  # (B,)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)# (B, obs_dim)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)      # (B,)

        # Current Q(s,a)
        q_all = self.q(obs)                                        # (B, A)
        q_sa = q_all.gather(1, actions.view(-1, 1)).squeeze(1)     # (B,)

        # Target: r + γ max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            q_next_all = self.target(next_obs)                     # (B, A)
            q_next_max = q_next_all.max(dim=1).values              # (B,)
            y = rewards + float(gamma) * (1.0 - dones) * q_next_max

        loss = nn.SmoothL1Loss()(q_sa, y)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=self.grad_clip_norm)
        self.optimizer.step()

        with torch.no_grad():
            stats = {
                "loss": float(loss.item()),
                "q_mean": float(q_sa.mean().item()),
                "target_mean": float(y.mean().item()),
            }
        return stats

    # -------------------- Target updates --------------------

    def update_target(self) -> None:
        """
        Polyak-averaged target update:
            θ' ← (1 - τ) θ' + τ θ
        If τ <= 0, perform a hard copy.
        """
        if self.tau <= 0:
            self._hard_update()
            return
        with torch.no_grad():
            for p_targ, p in zip(self.target.parameters(), self.q.parameters()):
                p_targ.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

    def _hard_update(self) -> None:
        """θ' ← θ (exact copy)."""
        self.target.load_state_dict(self.q.state_dict())

    # -------------------- Helpers --------------------

    def _mask_from_admissible(
        self, admissible: Union[np.ndarray, Sequence[int]]
    ) -> torch.Tensor:
        """
        Convert admissible set to boolean mask tensor on the correct device.
        Accepts a boolean array mask or a list of allowed indices.
        """
        if isinstance(admissible, np.ndarray) and admissible.dtype == bool:
            mask = torch.as_tensor(admissible, dtype=torch.bool, device=self.device)
            if mask.numel() != self.n_actions:
                # If wrong size, fall back to all True
                mask = torch.ones(self.n_actions, dtype=torch.bool, device=self.device)
            return mask

        # treat as indices
        idx = list(int(i) for i in admissible)
        mask = torch.zeros(self.n_actions, dtype=torch.bool, device=self.device)
        idx = [i for i in idx if 0 <= i < self.n_actions]
        if len(idx) == 0:
            mask[:] = True
        else:
            mask[idx] = True
        return mask

    def _allowed_indices_from_admissible(
        self, admissible: Union[np.ndarray, Sequence[int]]
    ) -> list[int]:
        """Return explicit list of allowed action indices."""
        if isinstance(admissible, np.ndarray) and admissible.dtype == bool:
            return [i for i, ok in enumerate(admissible.tolist()) if ok]
        return [int(i) for i in admissible if 0 <= int(i) < self.n_actions]
