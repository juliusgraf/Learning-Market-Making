# rl_auctions/env/orderbook.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import math
from collections import defaultdict

@dataclass
class OrderBookState:
    """Depth, volumes, mid, and standing orders for the CLOB phase."""
    mid_price: float
    depth_plus: int
    depth_minus: int
    levels_plus: List[int]   # V^{+,j}_t for j<=L_c (ask side, j=1 is best ask)
    levels_minus: List[int]  # V^{-,j}_t for j<=L_c (bid side, j=1 is best bid)
    standing_orders: List[Tuple[float, int]]  # O_i ⊆ (αZ)×N, after current action

class OrderBook:
    """
    Minimal CLOB to simulate arrivals, matching, and agent priority (Assumption 2).
    - We track exogenous per-level volumes V^{+,j}, V^{-,j} up to depth L_c.
    - The agent’s limit (sell) order sits at level j_a with strict priority at that level.
    - Hypothetical clearing price H_cl is estimated via Algorithm 1 running moments.

    References:
      • Execution with priority & volume formula (1).  Et̂i = max(0, min(v̂i, M^- - Σ_{j<j_i} V^{+,j}))  — Assumption 2.  # noqa
        (see paper, Eq. (1) and surrounding text)  # noqa
      • Bounded book depth L_c and per-level volumes definition.  # noqa
      • Algorithm 1 for H_cl with K̂^k_i and smoothing.  # noqa
    """

    def __init__(self, alpha: float, L_c: int, V_max: int):
        """
        Args:
            alpha: Tick size α.
            L_c: Maximum visible depth per side.
            V_max: Maximum per-level background volume used for initialization.
        """
        self.alpha = alpha
        self.L_c = int(L_c)
        self.V_max = int(V_max)

        # Exogenous book volumes (do NOT include the agent’s order)
        self._exog_plus: List[int] = [0] * self.L_c   # asks, j=1..L_c
        self._exog_minus: List[int] = [0] * self.L_c  # bids, j=1..L_c

        # Agent order (kept separate to enforce priority on matching)
        self._agent_level: Optional[int] = None  # j_a in {1..L_c}
        self._agent_volume: int = 0

        # State cache
        self.state: Optional[OrderBookState] = None

        # --- Algorithm 1 running statistics over standing orders O_i ---
        # For each grid index k (price = αk), maintain:
        #   sum_vol[k] = Σ_{s=1..i} (total outstanding volume at αk at step s)
        #   sum_vol_sq[k] = Σ_{s=1..i} (that total)^2
        # i.e., the “per-time” totals that appear in ê^k_i and ς̂^k_i in Alg. 1.
        self._alg1_sum_vol = defaultdict(float)
        self._alg1_sum_vol_sq = defaultdict(float)
        self._alg1_steps = 0  # i in Algorithm 1

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------
    def reset(self, mid_price: float) -> OrderBookState:
        """
        Reinitialize levels and standing orders at episode start.
        We seed a symmetric, non-empty book so that taker arrivals can match.
        """
        # Fill each level with a modest background volume (toy emulator choice).
        self._exog_plus = [self.V_max] * self.L_c
        self._exog_minus = [self.V_max] * self.L_c

        self._agent_level = None
        self._agent_volume = 0
        self._alg1_sum_vol.clear()
        self._alg1_sum_vol_sq.clear()
        self._alg1_steps = 0

        depth_p = self._first_zero_depth(self._exog_plus)
        depth_m = self._first_zero_depth(self._exog_minus)

        standing = self._build_standing_orders(mid_price)
        self.state = OrderBookState(
            mid_price=mid_price,
            depth_plus=depth_p,
            depth_minus=depth_m,
            levels_plus=list(self._exog_plus),
            levels_minus=list(self._exog_minus),
            standing_orders=standing,
        )
        return self.state

    def refresh_standing_orders_from_ladders(self, mid: float) -> None:
        """
        Rebuild state.standing_orders from current ladders V_plus/V_minus
        using the snapped grid mid.
        """
        if self.state is None:
            raise RuntimeError("OrderBook.reset must be called before refreshing orders.")

        grid_mid = round(mid / self.alpha) * self.alpha
        orders = []

        # asks: +1 .. +L_c
        for j, vol in enumerate(self.state.levels_plus, start=1):
            v = float(vol)
            if v > 0.0:
                price = grid_mid + j * self.alpha
                orders.append((price, v))

        # bids: -1 .. -L_c
        for j, vol in enumerate(self.state.levels_minus, start=1):
            v = float(vol)
            if v > 0.0:
                price = grid_mid - j * self.alpha
                orders.append((price, v))

        self.state.standing_orders = orders
    # ---------------------------------------------------------------------
    # Agent interaction (place limit sell)
    # ---------------------------------------------------------------------
    def apply_agent_limit(self, volume: int, delta_index: int) -> None:
        """
        Place the agent’s sell limit at price level j_a = delta_index (>0 means above mid).
        The agent has strict priority at j_a (Assumption 2).

        Args:
            volume: shares to sell (clipped to ≥0).
            delta_index: integer j offset from mid for the ask side (1 = best ask).
        """
        if self.state is None:
            raise RuntimeError("OrderBook.reset must be called before placing an order.")

        j = int(delta_index)
        if j < 1:
            j = 1
        if j > self.L_c:
            j = self.L_c

        self._agent_level = j
        self._agent_volume = max(0, int(volume))

        # Update the standing orders snapshot (include agent’s residual volume)
        self.state.standing_orders = self._build_standing_orders(self.state.mid_price)

    # ---------------------------------------------------------------------
    # Matching engine for taker arrivals during the continuous phase
    # ---------------------------------------------------------------------
    def simulate_taker_arrivals(self, buy_qty: int, sell_qty: int) -> Dict[str, Any]:
        """
        Apply taker market orders to the book and compute the agent’s execution E_t̂i.

        buy_qty  ≡ M^- (market buys hitting the ask side),
        sell_qty ≡ M^+ (market sells hitting the bid side).
        Execution with priority follows Eq. (1) in the paper.

        Returns:
            {
              "agent_exec": int,
              "buy_consumed": int,
              "sell_consumed": int,
            }
        """
        if self.state is None:
            raise RuntimeError("OrderBook.reset must be called before matching.")

        buy_qty = max(0, int(buy_qty))
        sell_qty = max(0, int(sell_qty))

        # --- Compute agent execution using priority rule (Eq. (1)) ---
        agent_exec = 0
        remaining_buy = buy_qty

        if self._agent_level is not None and self._agent_volume > 0:
            ja = self._agent_level
            # Sum of exogenous volumes strictly better (closer to mid) than agent’s level
            sum_prior = sum(self._exog_plus[: max(0, ja - 1)])
            potential = remaining_buy - sum_prior
            if potential > 0:
                agent_exec = min(self._agent_volume, potential)
            # We will actually decrement in the flow below to keep accounting consistent.

        # --- Consume asks with market buys (respecting agent priority at j_a) ---
        # 1) Levels strictly better than agent’s level
        for j in range(1, self.L_c + 1):
            if remaining_buy <= 0:
                break
            # Exogenous volume at this ask level
            v_exog = self._exog_plus[j - 1]

            if self._agent_level is not None and j == self._agent_level:
                # Agent gets priority at his level
                take_a = min(self._agent_volume, remaining_buy)
                self._agent_volume -= take_a
                remaining_buy -= take_a

                # Residual buy volume consumes exogenous at the same level
                if remaining_buy > 0 and v_exog > 0:
                    take_exog = min(v_exog, remaining_buy)
                    self._exog_plus[j - 1] -= take_exog
                    remaining_buy -= take_exog
            else:
                # No agent here; consume exogenous
                if v_exog > 0:
                    take_exog = min(v_exog, remaining_buy)
                    self._exog_plus[j - 1] -= take_exog
                    remaining_buy -= take_exog

        buy_consumed = buy_qty - remaining_buy

        # --- Consume bids with market sells (mirror side; agent unaffected) ---
        remaining_sell = sell_qty
        for j in range(1, self.L_c + 1):
            if remaining_sell <= 0:
                break
            v_exog = self._exog_minus[j - 1]
            if v_exog > 0:
                take_exog = min(v_exog, remaining_sell)
                self._exog_minus[j - 1] -= take_exog
                remaining_sell -= take_exog
        sell_consumed = sell_qty - remaining_sell

        # --- Update depths and standing orders snapshot ---
        depth_p = self._first_zero_depth(self._exog_plus)
        depth_m = self._first_zero_depth(self._exog_minus)
        self.state.depth_plus = depth_p
        self.state.depth_minus = depth_m
        self.state.levels_plus = list(self._exog_plus)
        self.state.levels_minus = list(self._exog_minus)
        self.state.standing_orders = self._build_standing_orders(self.state.mid_price)

        return {
            "agent_exec": agent_exec,
            "buy_consumed": int(buy_consumed),
            "sell_consumed": int(sell_consumed),
        }

    # ---------------------------------------------------------------------
    # Algorithm 1: Hypothetical clearing price estimator for CLOB
    # ---------------------------------------------------------------------
    def compute_hypo_clearing(self, gamma_smooth: float, prev_Hcl: float) -> float:
        """
        Algorithm 1 (CLOB): form \u007eS from current standing orders, then smooth:
            H_cl(i) = H_cl(i-1) + γ ( \u007eS(i) - H_cl(i-1) ).

        We maintain the running moments per price level αk:
          ê^k_i = (1/i) Σ_{s=1..i} V_k(s),
          ς̂^k_i = (1/i) Σ_{s=1..i} V_k(s)^2,
          K̂^k_i = (2 ê^k_i - ς̂^k_i / ê^k_i) α^{-1}  (defined when ê^k_i>0),
        then solve Σ_k K̂^k_i (αk - p) = 0 ⇒ p = (Σ_k K̂^k_i αk) / (Σ_k K̂^k_i).
        See Algorithm 1 in the paper.  # noqa
        """
        if self.state is None:
            raise RuntimeError("OrderBook.reset must be called before computing H_cl.")

        # Safety clamp
        gamma = float(max(0.0, min(1.0, gamma_smooth)))

        # Current standing orders O_i: aggregate by grid index k (price = αk)
        level_totals: Dict[int, float] = defaultdict(float)
        for price, v in self.state.standing_orders:
            if v <= 0:
                continue
            k = int(round(price / self.alpha))
            level_totals[k] += float(v)

        if not level_totals:
            # Remark 1.3 assumes O_i non-empty; fallback: keep previous H_cl if empty.  # noqa
            return float(prev_Hcl)

        # Update running moments (step i)
        self._alg1_steps += 1
        i = self._alg1_steps
        for k, tot in level_totals.items():
            self._alg1_sum_vol[k] += tot
            self._alg1_sum_vol_sq[k] += tot * tot

        # Compute K̂^k_i and weighted average for \u007eS
        num = 0.0
        den = 0.0
        for k, sum_v in self._alg1_sum_vol.items():
            e_hat = sum_v / i
            if e_hat <= 1e-12:
                continue
            s2_hat = self._alg1_sum_vol_sq[k] / i
            K_hat = (2.0 * e_hat - s2_hat / e_hat) / self.alpha  # Alg. 1 line 9
            if K_hat <= 0:
                # If degenerate/negative slope appears numerically, ignore this level
                continue
            price_k = self.alpha * k
            num += K_hat * price_k
            den += K_hat

        if den <= 1e-12:
            S_tilde = float(prev_Hcl)  # ill-conditioned; fallback
        else:
            S_tilde = num / den  # Alg. 1 line 11: solve Σ K̂(αk - p)=0

        # Smooth to get the new H_cl (Alg. 1 line 12)
        H_new = float(prev_Hcl + gamma * (S_tilde - prev_Hcl))
        return H_new

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _first_zero_depth(self, levels: List[int]) -> int:
        """
        Return the smallest j≥1 with V^{·,j}=0; if none in [1..L_c], return L_c+1.
        Mirrors the paper’s L^ζ_t̂i definition.
        """
        for j, v in enumerate(levels, start=1):
            if v <= 0:
                return j
        return self.L_c + 1

    def _build_standing_orders(self, mid_price: float) -> List[Tuple[float, int]]:
        """
        Build O_i ⊆ (αZ)×N after the current action, as required by Algorithm 1:
        include all *unexecuted* exogenous book volumes on both sides and the agent’s
        current residual (if any). Volumes are unsigned (positive) per the paper’s convention.
        """
        orders: List[Tuple[float, int]] = []

        # Bids (buy limits): prices below mid
        for j, v in enumerate(self._exog_minus, start=1):
            if v > 0:
                price = mid_price - j * self.alpha
                orders.append((price, int(v)))

        # Asks (sell limits): prices above mid
        for j, v in enumerate(self._exog_plus, start=1):
            if v > 0:
                price = mid_price + j * self.alpha
                orders.append((price, int(v)))

        # Agent’s residual sell order (priority at its level)
        if self._agent_level is not None and self._agent_volume > 0:
            price = mid_price + self._agent_level * self.alpha
            orders.append((price, int(self._agent_volume)))

        return orders
