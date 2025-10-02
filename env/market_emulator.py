from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .orderbook import OrderBook, OrderBookState
from .auction import Auction, AuctionState
from .clearing import ClearingPriceEstimator

# ---------------------------------------------------------------------
# Environment configuration & I/O structs
# ---------------------------------------------------------------------
@dataclass
class EnvConfig:
    # time structure
    tau_open: int = 50          # τ_op
    tau_close: int = 60         # τ_cl  (episode length = τ_cl)
    # pricing/grid
    alpha: float = 0.01         # tick size α
    initial_mid: float = 100.0  # S_mid,0
    # CLOB background
    L_c: int = 10               # max depth levels per side
    V_max: int = 50             # per-level background volume cap
    # takers in CLOB (latent)
    lambda_arrival: float = 1.5 # Poisson rate for total takers per step
    v_market: int = 5           # uniform marked volume v per taker
    # Algorithm 1 smoothing
    gamma_smooth: float = 0.4   # γ
    # Brownian mid dynamics (toy driftless)
    sigma_mid: float = 0.02     # σ per step
    # agent/inventory
    I_max: int = 500            # starting inventory (sell program)
    # CLOB reward
    kappa: float = 1.0          # penalty scaling; see κ=(nα)^(-1)
    # auction background (Example 1: linear makers + observable takers)
    L_a: int = 8                # La (max # exogenous makers kept)
    beta: float = 0.25          # β quantization for Ka grid
    eps_contract: float = 0.1   # ε in (1-ε)Kμ/La cap for exo makers (optional)
    # exogenous arrival parameters (auction)
    p_new_mm: float = 0.6
    p_cancel_mm: float = 0.25
    p_new_taker: float = 0.7
    p_cancel_taker: float = 0.15
    K_l: float = 0.05           # maker slope lower/upper bounds
    K_u: float = 1.50
    M_side_grid: int = 8        # placement radius in ticks around last H_cl
    pareto_vm: float = 1.0
    pareto_gamma: float = 1.5
    # auction rewards
    q_wrong_side: float = 1.0   # q
    d_cancel: float = 0.05      # d (per cancellation)
    lambda_term: float = 1e-3  # λ

@dataclass
class StepResult:
    next_state: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------
# MarketEmulator
# ---------------------------------------------------------------------
class MarketEmulator:
    """
    Two-phase environment:
      • CLOB on t ∈ {0, …, τ_op-1} with latent takers and Alg. 1 H_cl updates.
      • Auction on t ∈ {τ_op, …, τ_cl-1} with observable takers/makers,
        linear supplies, cancellations, and terminal clearing at t=τ_cl.

    State encoders/action spaces live in env/state_encoder.py and env/action_space.py.
    """

    def __init__(self, cfg: EnvConfig, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.orderbook = OrderBook(alpha=cfg.alpha, L_c=cfg.L_c, V_max=cfg.V_max)
        self.auction = Auction(alpha=cfg.alpha, beta=cfg.beta, L_a=cfg.L_a, eps_contract=cfg.eps_contract)
        self.hcl_est = ClearingPriceEstimator(alpha=cfg.alpha)

        # episode vars
        self.t: int = 0
        self.mid: float = cfg.initial_mid
        self.inventory: int = cfg.I_max
        self.H_cl: float = cfg.initial_mid
        self._last_S_cl: Optional[float] = None  # to warm-start Alg. 1 between episodes

        # phase states
        self.ob_state: Optional[OrderBookState] = None
        self.au_state: Optional[AuctionState] = None

        # bookkeeping
        self._I_tau_op: Optional[int] = None

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0
        self.mid = self.cfg.initial_mid
        self.inventory = self.cfg.I_max

        # Initialize H_cl,0 ← previous episode S_cl or mid (Alg. 1 requirement)
        self.H_cl = float(self._last_S_cl if self._last_S_cl is not None else self.mid)

        # Reset subsystems
        self.ob_state = self.orderbook.reset(mid_price=self.mid)
        h = max(1, self.cfg.tau_close - self.cfg.tau_open)
        self.au_state = self.auction.reset(mid_price=self.mid, I_tau_op=self.inventory, h=h)

        self._I_tau_op = None  # will set on the transition boundary

        return self.get_state_dict()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def current_phase(self) -> str:
        if self.t < self.cfg.tau_open:
            return "clob"
        elif self.t < self.cfg.tau_close:
            return "auction"
        return "terminal"

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "phase": self.current_phase(),
            "mid": self.mid,
            "H_cl": self.H_cl,
            "inventory": self.inventory,
            "orderbook": None if not self.ob_state else {
                "L_plus": self.ob_state.depth_plus,
                "L_minus": self.ob_state.depth_minus,
                "V_plus": list(self.ob_state.levels_plus),
                "V_minus": list(self.ob_state.levels_minus),
            },
            "auction": None if not self.au_state else {
                "k_t": self.au_state.k_t,
                "N_plus": self.au_state.N_plus,
                "N_minus": self.au_state.N_minus,
                "theta_sum": int(sum(self.au_state.theta)) if self.au_state.theta else 0,
                "H_cl": self.au_state.H_cl,
                "I_tau_op": self.au_state.I_tau_op,
            },
        }

    def step(self, action: Dict[str, Any]) -> StepResult:
        """
        action (structured):
          CLOB (t < τ_op): {'v': int, 'delta': int}
          Auction (τ_op ≤ t < τ_cl): {'K_a': float, 'S_a': float, 'c': List[int]}
        """
        assert self.current_phase() != "terminal", "Episode already finished."

        info: Dict[str, Any] = {"phase": self.current_phase()}
        reward = 0.0

        # ------------------ CLOB PHASE ------------------
        if self.current_phase() == "clob":
            v = int(max(0, action.get("v", 0)))
            delta_idx = int(action.get("delta", 1))
            # admissibility trims
            v = min(v, self.inventory, self.cfg.V_max)
            delta_idx = max(1, min(self.cfg.L_c, delta_idx))  # ask side j ≥ 1

            # Agent places order and we simulate takers
            self.orderbook.apply_agent_limit(volume=v, delta_index=delta_idx)

            buy_qty, sell_qty = self._sample_clob_takers()
            match_res = self.orderbook.simulate_taker_arrivals(buy_qty=buy_qty, sell_qty=sell_qty)
            agent_exec = int(match_res["agent_exec"])
            self.inventory = max(0, self.inventory - agent_exec)

            # Compute H_cl via Algorithm 1 (using standing orders snapshot)
            self.H_cl = self.orderbook.compute_hypo_clearing(self.cfg.gamma_smooth, prev_Hcl=self.H_cl)

            # Execution price S•_t = α(δ_mid + delta_idx), we snap mid to the α-grid
            grid_mid = round(self.mid / self.cfg.alpha) * self.cfg.alpha
            S_star = grid_mid + delta_idx * self.cfg.alpha

            # Reward per paper: S• * E * f(1 - κ f(H_cl - S•)), f(x)=(x)+
            reward = self._reward_clob(exec_qty=agent_exec, S_star=S_star, H_cl=self.H_cl)

            info.update({
                "S_star": S_star,
                "agent_exec": agent_exec,
                "buy_consumed": match_res["buy_consumed"],
                "sell_consumed": match_res["sell_consumed"],
                "delta_idx": delta_idx,
                "v_submitted": v,
                "H_cl": self.H_cl,
            })

            # evolve mid (toy)
            self._update_mid_brownian()

            # advance time; if we cross to auction, set I_tau_op and re-init auction state mid
            self.t += 1
            if self.t == self.cfg.tau_open:
                self._I_tau_op = int(self.inventory)
                # re-seed auction state with frozen inventory and current mid
                h = max(1, self.cfg.tau_close - self.cfg.tau_open)
                self.au_state = self.auction.reset(mid_price=self.mid, I_tau_op=self._I_tau_op, h=h)
                self.au_state.H_cl = self.H_cl  # carry current estimate as seed

            return StepResult(next_state=self.get_state_dict(), reward=float(reward), done=False, info=info)

        # ------------------ AUCTION PHASE ------------------
        assert self.au_state is not None
        Ka = float(max(0.0, action.get("K_a", 0.0)))
        Sa = float(action.get("S_a", self.mid))
        c_vec: List[int] = list(action.get("c", [0] * (self.cfg.tau_close - self.cfg.tau_open)))

        # Set agent order (quantization handled inside Auction)
        self.auction.set_agent_order(Ka, Sa, c_vec)

        # Apply cancellations into θ_t with feasibility ct+1 ≤ 1-θ_t
        cancels_applied = 0
        pending = self.auction.pending_cancel or []
        local_t = self.auction._t  # 0-based inside auction window
        if pending:
            for s in range(min(local_t, len(self.au_state.theta))):
                if pending[s] and self.au_state.theta[s] == 0:
                    self.au_state.theta[s] = 1
                    cancels_applied += 1
        self.auction.clear_pending_cancel()

        # Exogenous arrivals (makers & takers)
        self.auction.step_exogenous(self.au_state, params=dict(
            p_new_mm=self.cfg.p_new_mm,
            p_cancel_mm=self.cfg.p_cancel_mm,
            p_new_taker=self.cfg.p_new_taker,
            p_cancel_taker=self.cfg.p_cancel_taker,
            K_l=self.cfg.K_l, K_u=self.cfg.K_u,
            M_side_grid=self.cfg.M_side_grid,
            pareto_vm=self.cfg.pareto_vm, pareto_gamma=self.cfg.pareto_gamma,
            rng=self.rng
        ))

        # Update intra-auction clearing estimate H_cl (eq. (2) with linear supplies)
        H_now = self.auction.compute_H_cl(self.au_state)
        self.H_cl = H_now

        # One-step auction reward (non-terminal)
        reward = self._reward_auction_step(K_a=Ka, S_a=Sa, c_count=cancels_applied, H_cl=self.H_cl)

        # Time advance
        self.auction._t += 1
        self.t += 1

        done = False
        term_info = {}
        # If we just stepped from t=τ_cl-1 → t=τ_cl, compute terminal clearing & add terminal reward
        if self.t >= self.cfg.tau_close:
            S_cl, executed = self.auction.finalize_clearing(self.au_state)
            I_tau_cl = int((self._I_tau_op or 0) - executed)
            # Terminal reward as in paper (we add it on the final transition step)
            r_term = self._reward_terminal(
                S_cl=S_cl,
                executed=executed,
                I_tau_cl=I_tau_cl,
                K_hist=self.auction._agent_K,
                S_hist=self.auction._agent_S,
                theta=self.au_state.theta
            )
            reward += r_term
            self._last_S_cl = float(S_cl)  # warm-start next episode's Alg. 1
            done = True
            term_info = {"S_cl": S_cl, "executed_terminal": executed, "I_tau_cl": I_tau_cl, "r_terminal": r_term}

        info.update({
            "K_a": Ka, "S_a": Sa, "c_applied": cancels_applied,
            "theta_sum": int(sum(self.au_state.theta)),
            "k_makers": self.au_state.k_t,
            "N_plus": self.au_state.N_plus,
            "N_minus": self.au_state.N_minus,
            "H_cl": self.H_cl,
            **term_info
        })

        return StepResult(next_state=self.get_state_dict(), reward=float(reward), done=done, info=info)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _sample_clob_takers(self) -> Tuple[int, int]:
        """
        Paper models marked counting processes with uniform marks v, here we
        emulate arrivals with a Poisson total then split sides. M^ζ_t = v (N^ζ_t - N^ζ_{t-1}). :contentReference[oaicite:8]{index=8}
        """
        lam = max(0.0, self.cfg.lambda_arrival)
        n_total = self.rng.poisson(lam=lam)
        n_buy = self.rng.binomial(n_total, 0.5)
        n_sell = n_total - n_buy
        buy_qty = int(n_buy * self.cfg.v_market)
        sell_qty = int(n_sell * self.cfg.v_market)
        return buy_qty, sell_qty

    def _update_mid_brownian(self) -> None:
        """Driftless Gaussian increment for mid (toy)."""
        self.mid = float(self.mid + self.cfg.sigma_mid * self.rng.standard_normal())

    # ------------------- Rewards (paper-exact) -------------------
    @staticmethod
    def _f_plus(x: float) -> float:
        return x if x > 0 else 0.0

    def _reward_clob(self, exec_qty: int, S_star: float, H_cl: float) -> float:
        """
        r = S• * E * f(1 - κ f(H_cl - S•)), f(x)=(x)+. :contentReference[oaicite:9]{index=9}
        """
        gap = self._f_plus(H_cl - S_star)
        inner = 1.0 - self.cfg.kappa * gap
        factor = self._f_plus(inner)
        return float(S_star * exec_qty * factor)

    def _reward_auction_step(self, K_a: float, S_a: float, c_count: int, H_cl: float) -> float:
        """
        r = K_a (H_cl - S_a) - q f(-K_a (H_cl - S_a)) - d ||c_t||_1. :contentReference[oaicite:10]{index=10}
        """
        base = K_a * (H_cl - S_a)
        wrong = self.cfg.q_wrong_side * self._f_plus(-K_a * (H_cl - S_a))
        cancel = self.cfg.d_cancel * float(c_count)
        return float(base - wrong - cancel)

    def _reward_terminal(
        self,
        S_cl: float,
        executed: float,
        I_tau_cl: int,
        K_hist: List[float],
        S_hist: List[float],
        theta: List[int],
    ) -> float:
        """
        Terminal:
          Σ_s K_s (S_cl - S_s) (1 - θ^{(s)}_{τcl})  - λ |I_{τcl}|^2
          - q Σ_s f( - K_s (S_cl - S_s) (1 - θ^{(s)}_{τcl}) ). :contentReference[oaicite:11]{index=11}
        """
        term1 = 0.0
        wrong = 0.0
        for s, (Ks, Ss) in enumerate(zip(K_hist, S_hist)):
            mask = 1 - int(theta[s]) if s < len(theta) else 1
            qty = float(Ks) * (S_cl - float(Ss)) * float(mask)
            term1 += qty
            wrong += self._f_plus(-qty)

        penalty_inv = self.cfg.lambda_term * float(I_tau_cl ** 2)
        return float(term1 - penalty_inv - self.cfg.q_wrong_side * wrong)
