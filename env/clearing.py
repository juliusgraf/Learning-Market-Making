from typing import Dict, Tuple, List, Optional
from collections import defaultdict

class ClearingPriceEstimator:
    """
    Helper for:
      (i) Algorithm 1 (CLOB hypothetical clearing price H_cl with smoothing γ),
      (ii) auction fixed-point solves under linear supply (eqs. (2)–(3)).

    This class keeps the running moments across steps for Alg. 1:
      ê^k_i = (1/i) Σ_{s=1..i} V_k(s),
      ς̂^k_i = (1/i) Σ_{s=1..i} V_k(s)^2,
      K̂^k_i = (2 ê^k_i - ς̂^k_i / ê^k_i) α^{-1},
    and computes S̃_i by solving Σ_k K̂^k_i (αk − p) = 0, then
      H_cl(i) = H_cl(i-1) + γ (S̃_i − H_cl(i-1)).  (Algorithm 1)
    """
    def __init__(self, alpha: float):
        self.alpha = float(alpha)

        # Running moments over price grid indices k (price = αk)
        self._sum_vol = defaultdict(float)
        self._sum_vol_sq = defaultdict(float)
        self._steps = 0

    # ------------------------------------------------------------------
    # (i) CLOB Hypothetical clearing price (Algorithm 1)
    # ------------------------------------------------------------------
    def compute_hcl_continuous(self, standing_orders: List[Tuple[float, int]],
                               prev_hcl: float, gamma: float) -> float:
        """
        Update Algorithm 1 running statistics with the current O_i and
        return the next smoothed hypothetical clearing price H_cl.

        Args:
            standing_orders: list of (price, volume) pairs for O_i ⊆ (αZ)×N
            prev_hcl: previous value H_cl(i-1)
            gamma: smoothing γ ∈ (0,1]

        Returns:
            H_cl(i) after applying Algorithm 1 lines 11–12.
        """
        gamma = float(max(0.0, min(1.0, gamma)))

        # Aggregate current O_i per grid index k (price = αk)
        level_totals: Dict[int, float] = defaultdict(float)
        for price, vol in standing_orders:
            v = float(vol)
            if v <= 0:
                continue
            k = int(round(price / self.alpha))
            level_totals[k] += v

        if not level_totals:
            # Algorithm 1 assumes O_i non-empty; fall back to prev value.
            # (Remark 1.3)
            return float(prev_hcl)

        # Update running moments
        self._steps += 1
        i = self._steps
        for k, tot in level_totals.items():
            self._sum_vol[k] += tot
            self._sum_vol_sq[k] += tot * tot

        # Compute K̂^k_i and solve Σ_k K̂(αk − p)=0  ⇒ p = (Σ K̂ αk) / (Σ K̂)
        num = 0.0
        den = 0.0
        for k, sum_v in self._sum_vol.items():
            e_hat = sum_v / i
            if e_hat <= 1e-12:
                continue
            s2_hat = self._sum_vol_sq[k] / i
            K_hat = (2.0 * e_hat - s2_hat / e_hat) / self.alpha
            if K_hat <= 0:
                continue
            price_k = self.alpha * k
            num += K_hat * price_k
            den += K_hat

        if den <= 1e-12:
            s_tilde = float(prev_hcl)  # ill-conditioned; keep previous
        else:
            s_tilde = num / den

        # Smoothing step
        return float(prev_hcl + gamma * (s_tilde - prev_hcl))

    # ------------------------------------------------------------------
    # (ii) Auction clearing under linear supplies (eqs. (2)–(3))
    # ------------------------------------------------------------------
    def solve_linear_supply_clearing(self, terms: Dict[str, List]) -> float:
        """
        Solve the linear clearing equation for p:
            Σ_i K_i (p - S_i)        (exogenous makers)
          + Σ_s K^a_s (p - S^a_s)    (agent's active orders; caller applies masks)
          + Σ_ζ Σ_i ζ ν^ζ_i          (taker imbalance, ζ∈{-1,+1})
          = 0

        For linear supply, this reduces to:
            (Σ K_total) * p = (Σ K_i S_i + Σ K^a_s S^a_s) - (Σ_ζ Σ_i ζ ν^ζ_i)

        Input `terms` dictionary (lists can be empty):
            makers:        List[Tuple[K_i, S_i]]
            agent:         List[Tuple[K_a, S_a]]
            takers_plus:   List[float]   # ζ = +1 (sellers)
            takers_minus:  List[float]   # ζ = -1 (buyers)

        Returns:
            Clearing price p (float). If ΣK == 0, returns 0 (or caller can override).
        """
        makers: List[Tuple[float, float]] = terms.get("makers", [])
        agent:  List[Tuple[float, float]] = terms.get("agent", [])
        tplus:  List[float] = terms.get("takers_plus", []) or []
        tminus: List[float] = terms.get("takers_minus", []) or []

        # Accumulate slopes and slope-weighted references
        sum_K = 0.0
        sum_KS = 0.0

        for Ki, Si in makers:
            Ki = float(Ki); Si = float(Si)
            if Ki > 0:
                sum_K += Ki
                sum_KS += Ki * Si

        for Ka, Sa in agent:
            Ka = float(Ka); Sa = float(Sa)
            if Ka > 0:
                sum_K += Ka
                sum_KS += Ka * Sa

        # Taker imbalance term   Σ_ζ Σ_i ζ ν^ζ_i  (sellers +, buyers −)
        taker_imbalance = float(sum(tplus) - sum(tminus))

        if sum_K <= 1e-12:
            # No slope: equation is ill-posed; return neutral price 0.0 by default.
            # Caller may want to keep previous H_cl / mid instead.
            return 0.0

        p = (sum_KS - taker_imbalance) / sum_K
        return float(p)
