from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import numpy as np


class ActionSpace:
    """
    Discrete wrapper for the paper’s action sets (Definition 1.2):

      CLOB (sell program):
        - volume v ∈ {0,1,...,V_max}
        - price level index δ ∈ {1,...,B}   (ask side; 1 = best ask)

      Auction (linear supply; discretized):
        - Ka ∈ {0, β, 2β, ..., K_max·β}
        - Sa ∈ {S_ref + α·z : z ∈ {-B, ..., 0, ..., +B}}  (computed at decode time
             using the last seen reference price; see below)
        - ct ∈ CANCELLATION_PATTERNS  (maps to θ-updates; env clamps feasibility)

    Design notes:
      • We build ONE global list of discrete actions spanning both phases.
        - [0 : N_clob)     → CLOB actions
        - [N_clob : N_all) → Auction actions
      • `admissible(t, obs, constraints)`:
        - Caches minimal context from `obs` (mid, H_cl, t, tau_open) so that `decode`
          can convert auction offsets into absolute prices for Sa.
        - Returns the indices valid for the CURRENT phase (masking the other half).
        - Optionally filters impossible CLOB actions (v > inventory).
      • `decode(a_idx)` returns a structured dict with keys appropriate for the phase:
          {"v", "delta"} for CLOB or {"K_a", "S_a", "c"} for Auction.
        It uses cached context from the last `admissible` call to compute S_a.
    """

    # Simple cancellation pattern IDs
    _CANCELLATION_NONE = 0
    _CANCELLATION_LAST = 1
    # >= 2: cancel at absolute index s (0-based), i.e. id = 2 + s

    def __init__(self, V_max: int, B: int, beta: float, K_max: int, h: int,
                 alpha: Optional[float] = None, S_anchor: Optional[float] = None):
        """
        Args:
            V_max: Max volume per CLOB action.
            B: Half-width (in ticks) around the reference for price indices.
            beta: Ka quantization step.
            K_max: Max multiple for Ka grid (inclusive).
            h: Auction horizon length (controls cancellation vector length).
            alpha: (optional) tick size for converting Sa offsets into prices.
                   If None, defaults to 0.01. You can set later via `set_alpha`.
            S_anchor: (optional) default reference price to start with; if None,
                      defaults to 100.0 until `admissible` sees a state.
        """
        self.V_max = int(V_max)
        self.B = int(B)
        self.beta = float(beta)
        self.K_max = int(K_max)
        self.h = int(max(1, h))

        # Price conversion context (can be updated from `admissible`)
        self.alpha = float(alpha if alpha is not None else 0.01)
        self.S_ref = float(S_anchor if S_anchor is not None else 100.0)
        self._last_local_t = 0  # auction-local time ≈ max(0, t - tau_open)

        # --- Enumerate the global discrete action set ---
        self._clob_actions: List[Tuple[int, int]] = []           # (v, delta_idx)
        self._auction_actions: List[Tuple[int, int, int]] = []   # (Ka_idx, Sa_offset_idx, cancel_id)

        # CLOB: v ∈ [0..V_max], δ ∈ [1..B]
        for v in range(0, self.V_max + 1):
            for d in range(1, self.B + 1):
                self._clob_actions.append((v, d))

        # Auction:
        #   Ka_idx ∈ [0..K_max]  -> Ka = beta * Ka_idx
        #   Sa_offset_idx ∈ [-B..B]  -> Sa = S_ref + alpha * offset
        #   cancel_id ∈ {0 (none), 1 (last), 2..(2+h-1)=cancel at s}
        cancel_ids = [self._CANCELLATION_NONE, self._CANCELLATION_LAST] + [2 + s for s in range(self.h)]
        for k_idx in range(0, self.K_max + 1):
            for z in range(-self.B, self.B + 1):
                for c_id in cancel_ids:
                    self._auction_actions.append((k_idx, z, c_id))

        self._N_clob = len(self._clob_actions)
        self._N_auction = len(self._auction_actions)
        self._N_total = self._N_clob + self._N_auction

        # Cache of the last observed raw state (for decode)
        self._last_obs: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Optional setters
    # ------------------------------------------------------------------
    def set_alpha(self, alpha: float) -> None:
        """Update tick size used to convert auction offsets to prices."""
        self.alpha = float(alpha)

    def set_reference_price(self, price: float) -> None:
        """Update the anchor price used for auction Sa decoding."""
        self.S_ref = float(price)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def size(self) -> int:
        """Total number of discrete actions across both phases."""
        return self._N_total

    def admissible(self, t: int, obs: Dict, constraints: Dict) -> List[int]:
        """
        Compute indices allowed given the current phase and basic constraints.

        Inputs:
            t: global time index from the environment
            obs: raw state dict (MarketEmulator.get_state_dict())
            constraints: may include 'inventory', 'tau_open', 'tau_close', 'phase'

        Returns:
            List of action indices (subset of [0..size()-1]) that are admissible now.
        """
        phase = str(constraints.get("phase") or obs.get("phase") or "clob")
        inv = int(constraints.get("inventory") or obs.get("inventory") or 0)
        tau_open = int(constraints.get("tau_open") or 0)
        tau_close = int(constraints.get("tau_close") or 0)

        # ---- Cache context for subsequent decode() calls ----
        self._last_obs = obs
        self.S_ref = float(obs.get("H_cl") or obs.get("mid") or self.S_ref)
        # Auction-local time (0-based inside [tau_open .. tau_close-1])
        self._last_local_t = max(0, t - tau_open)

        if phase == "clob":
            # Filter CLOB actions that exceed remaining inventory
            idxs = []
            for i, (v, d) in enumerate(self._clob_actions):
                if v <= max(0, inv):
                    idxs.append(i)  # offset 0
            return idxs

        elif phase == "auction":
            # All auction actions admissible; env will clamp cancellations feasibly.
            # Return indices shifted by N_clob.
            return list(range(self._N_clob, self._N_total))

        # Fallback: if phase unknown, allow everything (not ideal, but safe)
        return list(range(self._N_total))

    def decode(self, a_idx: int) -> Dict[str, int | float | List[int]]:
        """
        Map a discrete index into a structured action dict.
        Uses the last cached context (from admissible()) to resolve auction Sa.

        Returns:
            For CLOB index:
                {"v": int, "delta": int}
            For Auction index:
                {"K_a": float, "S_a": float, "c": List[int] (len=h)}
        """
        if not (0 <= a_idx < self._N_total):
            raise IndexError(f"Action index {a_idx} out of range [0, {self._N_total-1}]")

        if a_idx < self._N_clob:
            v, d = self._clob_actions[a_idx]
            return {"v": int(v), "delta": int(d)}

        # Auction action
        a2 = a_idx - self._N_clob
        k_idx, z, c_id = self._auction_actions[a2]

        Ka = float(k_idx) * self.beta

        # Resolve Sa using the last cached reference; if none, fall back to default anchor
        S_ref = float(self.S_ref)
        alpha = float(self.alpha)
        Sa = S_ref + alpha * float(z)

        # Build cancellation vector c (length h)
        c_vec = [0] * self.h
        if c_id == self._CANCELLATION_NONE:
            pass
        elif c_id == self._CANCELLATION_LAST:
            # Cancel the most recent index < local_t (best effort; env ensures feasibility)
            s_last = max(0, min(self.h - 1, self._last_local_t - 1))
            if self._last_local_t > 0:
                c_vec[s_last] = 1
        else:
            # cancel at absolute index s = c_id - 2 (clamped)
            s = max(0, min(self.h - 1, c_id - 2))
            c_vec[s] = 1

        return {"K_a": Ka, "S_a": Sa, "c": c_vec}
