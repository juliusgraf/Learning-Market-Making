from __future__ import annotations

from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    """
    Convenience plotting utilities:
      - per-episode regret & cumulative regret
      - episode diagnostics: prices (mid/H_cl/quotes), inventory, rewards

    Usage:
        p = Plotter()
        p.plot_regret(regrets)
        p.savefig("runs/regret.png")

        p.plot_episode(traces)
        p.savefig("runs/episode_12.png")
    """

    def __init__(self, figsize=(9, 5)):
        self.figsize = figsize
        self._last_fig: Optional[plt.Figure] = None

    # ------------------------------------------------------------------

    def plot_regret(self, regrets: List[float]) -> None:
        """
        Plot per-episode regret (bars) and cumulative regret (line).

        Args:
            regrets: list of r_e values, length E
        """
        if regrets is None or len(regrets) == 0:
            # Create an empty fig to keep API consistent
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.set_title("Regret (no data)")
            ax.axis("off")
            self._last_fig = fig
            return

        r = np.asarray(regrets, dtype=float)
        cum = np.cumsum(r)
        x = np.arange(1, len(r) + 1)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.bar(x, r, alpha=0.5, label="Per-episode regret")
        ax.plot(x, cum, linewidth=2.0, label="Cumulative regret")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Regret")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")
        ax.set_title("Regret over episodes")
        self._last_fig = fig
        plt.tight_layout()

    # ------------------------------------------------------------------

    def plot_episode(self, traces: Dict[str, List[float]]) -> None:
        """
        Plot episode diagnostics. Expected (optional) keys in `traces`:

            "mid"          : list[float]  mid price
            "H_cl"         : list[float]  hypothetical/auction clearing estimate
            "S_star"       : list[float]  executed CLOB price per step (if any)
            "S_cl"         : float        terminal clearing price (single value)
            "inventory"    : list[float]  inventory path
            "reward"       : list[float]  per-step rewards

        Any missing key is skipped gracefully.
        """
        T = 0
        for k in ("mid", "H_cl", "S_star", "inventory", "reward"):
            if k in traces and isinstance(traces[k], list):
                T = max(T, len(traces[k]))
        x = np.arange(T)

        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 1.6), sharex=True)

        # --- Prices panel ---
        ax0 = axes[0]
        if "mid" in traces and len(traces["mid"]) > 0:
            ax0.plot(x, traces["mid"][:T], label="mid", linewidth=1.8)
        if "H_cl" in traces and len(traces["H_cl"]) > 0:
            ax0.plot(x, traces["H_cl"][:T], label="H_cl", linewidth=1.6)
        if "S_star" in traces and len(traces["S_star"]) > 0:
            ax0.plot(x, traces["S_star"][:T], label="S• (CLOB exec)", linewidth=1.2, linestyle=":")
        if "S_cl" in traces and traces.get("S_cl") is not None:
            s_cl = float(traces["S_cl"])
            ax0.axhline(s_cl, color="k", linestyle="--", linewidth=1.0, label="S_cl (terminal)")
        ax0.set_ylabel("Price")
        ax0.set_title("Prices")
        ax0.grid(True, linestyle="--", alpha=0.3)
        ax0.legend(loc="best")

        # --- Inventory panel ---
        ax1 = axes[1]
        if "inventory" in traces and len(traces["inventory"]) > 0:
            ax1.plot(x, traces["inventory"][:T], linewidth=1.8)
        ax1.set_ylabel("Inventory")
        ax1.set_title("Inventory path")
        ax1.grid(True, linestyle="--", alpha=0.3)

        # --- Reward panel ---
        ax2 = axes[2]
        if "reward" in traces and len(traces["reward"]) > 0:
            r = np.asarray(traces["reward"][:T], dtype=float)
            ax2.plot(x, r, label="reward", linewidth=1.2)
            ax2.plot(x, np.cumsum(r), label="cum. reward", linewidth=1.8)
            ax2.legend(loc="best")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("r / Σr")
        ax2.set_title("Rewards")
        ax2.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        self._last_fig = fig

    # ------------------------------------------------------------------

    def savefig(self, path: str) -> None:
        """Save the most recently created figure."""
        if self._last_fig is None:
            # Create an empty figure so the call doesn't fail silently
            fig, _ = plt.subplots(figsize=self.figsize)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return
        self._last_fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(self._last_fig)
        self._last_fig = None
