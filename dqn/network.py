from __future__ import annotations

from typing import Tuple, Iterable, Optional
import torch
import torch.nn as nn


def _make_mlp(
    input_dim: int,
    hidden_sizes: Iterable[int],
    activation: nn.Module,
    layer_norm: bool = False,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Utility: build an MLP stack (without the output layer)."""
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    return nn.Sequential(*layers)


def _init_weights(module: nn.Module, last_layer_small: bool = False) -> None:
    """
    Weight init: Kaiming for hidden layers (ReLU-friendly).
    Optionally shrink the final linear layer for stability.
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity="relu")
        nn.init.zeros_(module.bias)
        if last_layer_small:
            # Scale down to keep Q-values initially near zero
            with torch.no_grad():
                module.weight.mul_(0.01)
                module.bias.mul_(0.01)


class DQN(nn.Module):
    """
    Feedforward Q-network: f_θ(s) -> R^{|A|}

    Notes:
    - Your StateEncoder already includes time/phase features (finite-horizon),
      so we keep this strictly feedforward (no RNN).
    - If you enable `dueling=True`, we split into value/advantage streams:
        Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        *,
        dueling: bool = False,
        layer_norm: bool = False,
        dropout: float = 0.0,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert obs_dim > 0 and n_actions > 0, "obs_dim and n_actions must be positive"

        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.dueling = bool(dueling)

        act = activation if activation is not None else nn.ReLU()

        # Shared trunk (no output layer here)
        self.trunk = _make_mlp(
            input_dim=self.obs_dim,
            hidden_sizes=self.hidden_sizes,
            activation=act,
            layer_norm=layer_norm,
            dropout=dropout,
        )

        last_h = self.hidden_sizes[-1] if self.hidden_sizes else self.obs_dim

        if self.dueling:
            # Value stream V(s) -> R
            self.value_head = nn.Linear(last_h, 1)
            # Advantage stream A(s,a) -> R^{|A|}
            self.adv_head = nn.Linear(last_h, self.n_actions)
            # Init
            self.apply(lambda m: _init_weights(m, last_layer_small=False))
            _init_weights(self.value_head, last_layer_small=True)
            _init_weights(self.adv_head, last_layer_small=True)
        else:
            # Single head to Q-values directly
            self.head = nn.Linear(last_h, self.n_actions)
            # Init
            self.apply(lambda m: _init_weights(m, last_layer_small=False))
            _init_weights(self.head, last_layer_small=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, obs_dim) or (obs_dim,) — will be batched.

        Returns:
            Q-values of shape (B, n_actions), dtype=float32
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z = self.trunk(x)

        if self.dueling:
            v = self.value_head(z)                     # (B, 1)
            a = self.adv_head(z)                       # (B, A)
            a_centered = a - a.mean(dim=1, keepdim=True)
            q = v + a_centered
        else:
            q = self.head(z)

        return q.float()
