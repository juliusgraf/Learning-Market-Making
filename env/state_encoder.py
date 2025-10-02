from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

class StateEncoder:
    """
    Encodes the environment's structured state dict -> flat float32 vector for DQN.
    Produces a fixed-size vector independent of phase ("clob" or "auction").

    Feature layout (all float32):
      Common (7):
        [0]  t_norm = t / tau_close
        [1]  is_clob ∈ {0,1}
        [2]  is_auction ∈ {0,1}
        [3]  mid
        [4]  H_cl
        [5]  (H_cl - mid) / alpha   # price gap in ticks
        [6]  inventory

      CLOB-specific (2 + 2*L_c):
        [7]      depth_plus_norm  = min(L_plus, L_c+1) / (L_c+1)
        [8]      depth_minus_norm = min(L_minus, L_c+1) / (L_c+1)
        [9:9+L_c)         V_plus (len L_c; zeros if None)
        [9+L_c:9+2L_c)    V_minus (len L_c; zeros if None)

      Auction-specific (6):
        [9+2L_c]          k_t
        [10+2L_c]         N_plus
        [11+2L_c]         N_minus
        [12+2L_c]         theta_sum
        [13+2L_c]         I_tau_op (frozen inventory at auction start; 0 if N/A)
        [14+2L_c]         t_auction_norm = max(0, t - tau_open) / max(1, tau_close - tau_open)

    Total dimension = 15 + 2*L_c
    """

    def __init__(self, tau_close: int, alpha: float, B: int, L_c: int, L_a: int):
        # Note: B and L_a are not strictly required by this encoder, but kept for API consistency
        self.tau_close = int(tau_close)
        self.alpha = float(alpha)
        self.B = int(B)
        self.L_c = int(L_c)
        self.L_a = int(L_a)

        # Pre-compute obs dimension
        self._obs_dim = 15 + 2 * self.L_c

        # Placeholders for time boundaries; can be updated by caller if needed
        self._tau_open = None
        self._tau_close = self.tau_close

    def obs_dim(self) -> int:
        """Return the observation dimensionality given configured depths."""
        return self._obs_dim

    # Optional: allow wrapper to inform tau_open/close (for t_auction_norm)
    def set_horizon(self, tau_open: int, tau_close: int) -> None:
        self._tau_open = int(tau_open)
        self._tau_close = int(tau_close)

    # ------------------------------------------------------------------

    def encode(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Flatten/normalize the structured state dict into a fixed-size vector.
        Missing phase-specific fields are filled with zeros.

        Args:
            state: dict returned by MarketEmulator.get_state_dict()

        Returns:
            obs: np.ndarray (float32) with shape (obs_dim,)
        """
        t = int(state.get("t", 0))
        phase = str(state.get("phase", "clob"))
        mid = float(state.get("mid", 0.0))
        hcl = float(state.get("H_cl", mid))
        inv = float(state.get("inventory", 0.0))

        # Common block
        t_norm = float(t) / float(max(1, self._tau_close))  # default to ctor tau_close
        is_clob = 1.0 if phase == "clob" else 0.0
        is_auction = 1.0 if phase == "auction" else 0.0
        gap_ticks = (hcl - mid) / self.alpha if self.alpha > 0 else 0.0

        common = np.array(
            [t_norm, is_clob, is_auction, mid, hcl, gap_ticks, inv],
            dtype=np.float32,
        )

        # CLOB block
        ob = state.get("orderbook") or {}
        Lp = int(ob.get("L_plus", 0) or 0)
        Lm = int(ob.get("L_minus", 0) or 0)
        # Normalize depths to [0,1] by (L_c+1)
        denom_depth = float(self.L_c + 1)
        depth_plus_norm = float(min(max(Lp, 0), self.L_c + 1)) / denom_depth
        depth_minus_norm = float(min(max(Lm, 0), self.L_c + 1)) / denom_depth

        V_plus = np.zeros(self.L_c, dtype=np.float32)
        V_minus = np.zeros(self.L_c, dtype=np.float32)
        v_p_src = ob.get("V_plus") or []
        v_m_src = ob.get("V_minus") or []
        # Copy up to L_c entries; ignore overflow
        for i in range(min(self.L_c, len(v_p_src))):
            V_plus[i] = float(v_p_src[i])
        for i in range(min(self.L_c, len(v_m_src))):
            V_minus[i] = float(v_m_src[i])

        clob_block = np.concatenate(
            [np.array([depth_plus_norm, depth_minus_norm], dtype=np.float32), V_plus, V_minus],
            dtype=np.float32,
        )

        # Auction block
        au = state.get("auction") or {}
        k_t = float(au.get("k_t", 0.0) or 0.0)
        N_plus = float(au.get("N_plus", 0.0) or 0.0)
        N_minus = float(au.get("N_minus", 0.0) or 0.0)
        theta_sum = float(au.get("theta_sum", 0.0) or 0.0)
        I_tau_op = float(au.get("I_tau_op", 0.0) or 0.0)
        # Auction-relative normalized time
        tau_open = self._tau_open if self._tau_open is not None else 0
        tau_close = self._tau_close
        denom = max(1, tau_close - tau_open)
        t_auction_norm = float(max(0, t - tau_open)) / float(denom)

        auction_block = np.array(
            [k_t, N_plus, N_minus, theta_sum, I_tau_op, t_auction_norm],
            dtype=np.float32,
        )

        obs = np.concatenate([common, clob_block, auction_block], dtype=np.float32)

        # Safety: enforce fixed size
        if obs.shape[0] != self._obs_dim:
            # pad or truncate defensively (shouldn't happen with current design)
            if obs.shape[0] < self._obs_dim:
                pad = np.zeros(self._obs_dim - obs.shape[0], dtype=np.float32)
                obs = np.concatenate([obs, pad], dtype=np.float32)
            else:
                obs = obs[: self._obs_dim].astype(np.float32, copy=False)

        return obs