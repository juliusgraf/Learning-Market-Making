from __future__ import annotations

from typing import List, Dict, Any, Optional


class RegretAnalyzer:
    """
    Episodic regret tracker.

    Definitions
    -----------
    - Per-episode regret:
        r_e = V^*_0(e) - V^{π}_0(e)
      where V^*_0(e) is the GLFT benchmark value at t=0 for episode e,
      and V^{π}_0(e) is the realized return (proxy) from the learned policy.

    - Cumulative regret after E episodes:
        R_E = Σ_{e=1..E} r_e

    Usage
    -----
        ra = RegretAnalyzer()
        ra.add_episode(v_star_0, v_pi_0)
        total = ra.cumulative()
        hist = ra.history()
    """

    def __init__(self):
        self._regrets: List[float] = []
        self._cumulative: float = 0.0

    def add_episode(self, v_star_0: float, v_pi_0: float) -> None:
        """Record regret for one episode."""
        r = float(v_star_0) - float(v_pi_0)
        self._regrets.append(r)
        self._cumulative += r

    def cumulative(self) -> float:
        """Return Σ regrets so far."""
        return float(self._cumulative)

    def history(self) -> List[float]:
        """Return a copy of per-episode regrets [r_1, ..., r_E]."""
        return list(self._regrets)

    # Optional conveniences
    def mean(self) -> Optional[float]:
        """Return mean regret, or None if empty."""
        if not self._regrets:
            return None
        return float(sum(self._regrets) / len(self._regrets))

    def reset(self) -> None:
        """Clear all stored regrets."""
        self._regrets.clear()
        self._cumulative = 0.0

