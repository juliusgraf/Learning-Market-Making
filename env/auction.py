from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

@dataclass
class AuctionState:
    """Holds all auction-phase objects (t ≥ τ_op) per paper."""
    k_t: int                              # # auction limit orders (≤ L_a)
    N_plus: int
    N_minus: int
    theta: List[int]                      # cancellation mask θ_t ∈ {0,1}^h
    supply_funcs: Dict[int, Tuple[float, float]]  # i -> (K_i,t, S_i,t) linear
    taker_volumes_plus: List[float]       # ν_{+,i,·} (selling market orders)
    taker_volumes_minus: List[float]      # ν_{-,i,·} (buying market orders)
    H_cl: float                           # H_cl,t (auction-phase definition)
    I_tau_op: int                         # frozen inventory in auction


class Auction:
    """
    Auction module managing makers/takers, cancellations, and clearing.
    - Agent supply is linear Σ_t(p) = K^a_t (p - S^a_t) (Assumption 3).
    - Cancellations accumulate in θ_t; only (1 - θ^{(s)}_t) active orders remain.
    - H_cl,t solves Eq. (2); terminal S_cl solves Eq. (3) (linear case closed-form).
    """
    def __init__(self, alpha: float, beta: float, L_a: int, eps_contract: float):
        self.alpha = float(alpha)
        self.beta = float(beta)              # quantization for K_a
        self.L_a = int(L_a)                  # max # exogenous makers kept
        self.eps_contract = float(eps_contract)

        # Internal time index for the auction horizon [0..h-1] (τ_op..τ_cl-1)
        self._t: int = 0
        self._h: int = 0

        # Agent orders history (indexed by local auction time s)
        self._agent_K: List[float] = []
        self._agent_S: List[float] = []

        # For reproducibility, rely on external RNG if provided in step_exogenous params

        # Track last mid to place new exogenous S_i near it (for emulator convenience)
        self._last_mid: float = 0.0

        # Next id for exogenous makers
        self._next_maker_id: int = 0

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------
    def reset(self, mid_price: float, I_tau_op: int, h: int) -> AuctionState:
        """
        Reset auction state; initialize θ_{τ_op-1}=0 and empty books.
        Inventory remains frozen during auction (I_t = I_{τ_op}). :contentReference[oaicite:6]{index=6}
        """
        self._t = 0
        self._h = int(h)
        self._last_mid = float(mid_price)
        self._next_maker_id = 0

        self._agent_K = [0.0] * self._h
        self._agent_S = [0.0] * self._h

        state = AuctionState(
            k_t=0,
            N_plus=0,
            N_minus=0,
            theta=[0] * self._h,               # θ_{τ_op-1}=0, will accumulate with c_t. :contentReference[oaicite:7]{index=7}
            supply_funcs={},                   # i -> (K_i, S_i)
            taker_volumes_plus=[],             # sellers ν_{+,i,·}
            taker_volumes_minus=[],            # buyers  ν_{-,i,·}
            H_cl=mid_price,                    # initialize with mid as a sane seed
            I_tau_op=int(I_tau_op),
        )
        return state

    # ---------------------------------------------------------------------
    # Exogenous arrivals / cancels (emulator choices)
    # ---------------------------------------------------------------------
    def step_exogenous(self, state: AuctionState, params: Dict[str, float]) -> None:
        """
        Sample Bt (new maker), Dt (cancel), J±_t (takers), update sets/volumes.

        Expected `params` keys (emulator hyperparams):
          - 'p_new_mm', 'p_cancel_mm' : maker birth/cancel Bernoulli probs
          - 'p_new_taker', 'p_cancel_taker' : taker add/cancel probs (each side)
          - 'K_l', 'K_u' : range for maker slope K_i
          - 'M_side_grid' : integer grid radius to place S_i near mid (±M * α)
          - 'pareto_vm', 'pareto_gamma' : taker volume Pareto(scale, shape)
          - Optional: 'rng' (numpy Generator), 'K_mu' (lower bound used in contract)
        """
        rng: np.random.Generator = params.get("rng") or np.random.default_rng()

        # --- Maker addition ---
        if rng.random() < float(params.get("p_new_mm", 0.0)):
            if state.k_t < self.L_a:
                K_l = float(params.get("K_l", 0.1))
                K_u = float(params.get("K_u", 1.0))
                Ki = float(rng.uniform(K_l, K_u))

                # Respect Example 1 / contraction-style cap: Ki ≤ (1-ε) K_mu / L_a if provided. :contentReference[oaicite:8]{index=8}
                K_mu = params.get("K_mu", None)
                if K_mu is not None:
                    Ki = min(Ki, (1.0 - self.eps_contract) * float(K_mu) / float(self.L_a))

                # Place maker's reference S_i on α-grid around last mid
                M = int(params.get("M_side_grid", 5))
                z = int(rng.integers(-M, M + 1))
                Si = self._last_mid + z * self.alpha

                maker_id = self._next_maker_id
                self._next_maker_id += 1
                state.supply_funcs[maker_id] = (Ki, Si)
                state.k_t = len(state.supply_funcs)

        # --- Maker cancellation (remove a random maker) ---
        if rng.random() < float(params.get("p_cancel_mm", 0.0)) and state.supply_funcs:
            maker_id = rng.choice(list(state.supply_funcs.keys()))
            state.supply_funcs[maker_id] = (0, 0)
            state.k_t = len(state.supply_funcs)

        # --- Taker arrivals on each side (observed volumes) ---
        def pareto_vol():
            # Pareto(scale=vm, shape=γ): vm / U^{1/γ}
            vm = float(params.get("pareto_vm", 1.0))
            gamma = float(params.get("pareto_gamma", 1.5))
            u = rng.random()
            return float(vm / (u ** (1.0 / gamma)))

        # Add new takers
        if rng.random() < float(params.get("p_new_taker", 0.0)):
            v_plus = pareto_vol()  # seller (ζ=+1)
            state.taker_volumes_plus.append(v_plus)
            state.N_plus = len(state.taker_volumes_plus)

        if rng.random() < float(params.get("p_new_taker", 0.0)):
            v_minus = pareto_vol()  # buyer (ζ=-1)
            state.taker_volumes_minus.append(v_minus)
            state.N_minus = len(state.taker_volumes_minus)

        # Randomly cancel one historical taker on each side (set to zero in hindsight). :contentReference[oaicite:9]{index=9}
        if rng.random() < float(params.get("p_cancel_taker", 0.0)) and state.taker_volumes_plus:
            idx = int(rng.integers(0, len(state.taker_volumes_plus)))
            state.taker_volumes_plus[idx] = 0.0
        if rng.random() < float(params.get("p_cancel_taker", 0.0)) and state.taker_volumes_minus:
            idx = int(rng.integers(0, len(state.taker_volumes_minus)))
            state.taker_volumes_minus[idx] = 0.0

    # ---------------------------------------------------------------------
    # Agent action: (K_a, S_a, c_t)
    # ---------------------------------------------------------------------
    def set_agent_order(self, K_a: float, S_a: float, cancel_vec: List[int]) -> None:
        """
        Apply agent’s (K_a, S_a, c_t); update θ_t with constraint c_{t+1} ≤ 1 - θ_t. :contentReference[oaicite:10]{index=10}
        - K_a is quantized on the β grid and clipped to ≥0.
        - θ_t accumulates cancellations; only affects *past* orders.
        """
        # Quantize slope to β·N and ensure nonnegative
        K_q = max(0.0, round(float(K_a) / self.beta) * self.beta)
        S_q = float(S_a)

        # Record the current order at index s = t (local auction time)
        if 0 <= self._t < self._h:
            self._agent_K[self._t] = K_q
            self._agent_S[self._t] = S_q

        # Update θ_t = θ_{t-1} + c_t; enforce {0,1} with ct+1 ≤ 1 - θ_t
        # (the caller ensures feasibility across steps; here we just clamp)
        # θ is maintained inside AuctionState; we update via a reference later in env.MarketEmulator
        # NOTE: state is not passed here by design; cancellation cost is accounted in reward.
        self._pending_cancel = [int(min(1, max(0, x))) for x in cancel_vec]  # stash for env to apply

    # ---------------------------------------------------------------------
    # Clearing price estimators/solvers
    # ---------------------------------------------------------------------
    def compute_H_cl(self, state: AuctionState) -> float:
        """
        Solve Eq. (2) for H_cl,t under linear supplies gi,t(p)=K_i(p-S_i):
            [Σ_i K_i + Σ_s (1-θ^{(s)}_t) K^a_s] * p
          =  Σ_i K_i S_i + Σ_s (1-θ^{(s)}_t) K^a_s S^a_s  -  (Σ_ζ Σ_i ζ ν^{ζ}_i).
        Return the closed-form p. :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}
        """
        # Makers (current set)
        sum_K_m = 0.0
        sum_KS_m = 0.0
        for (Ki, Si) in state.supply_funcs.values():
            sum_K_m += float(Ki)
            sum_KS_m += float(Ki) * float(Si)

        # Agent (past orders up to t-1, masked by θ_t)
        sum_K_a = 0.0
        sum_KS_a = 0.0
        for s in range(0, min(self._t, self._h)):
            mask = 1 - int(state.theta[s])  # (1 - θ^{(s)}_t)
            if mask <= 0:
                continue
            Ks = float(self._agent_K[s])
            Ss = float(self._agent_S[s])
            sum_K_a += mask * Ks
            sum_KS_a += mask * Ks * Ss

        # Takers (observed volumes up to t-1): imbalance term Σ_ζ Σ_i ζ ν^{ζ}_i
        taker_imbalance = 0.0
        taker_imbalance += +sum(float(v) for v in state.taker_volumes_plus)   # ζ=+1 sellers
        taker_imbalance += -sum(float(v) for v in state.taker_volumes_minus)  # ζ=-1 buyers

        den = sum_K_m + sum_K_a
        if den <= 1e-12:
            # Ill-posed: no slope; keep previous H_cl
            return float(state.H_cl)

        num = (sum_KS_m + sum_KS_a) - taker_imbalance
        p = num / den
        state.H_cl = float(p)
        self._last_mid = float(p)  # reuse as "center" for new maker placements
        return float(p)

    def finalize_clearing(self, state: AuctionState) -> Tuple[float, float]:
        """
        At t=τ_cl: solve Eq. (3) for S_cl and compute executed agent volume:
            [Σ_i K_i + Σ_s (1-θ^{(s)}_{τcl}) K^a_s] * p
          =  Σ_i K_i S_i + Σ_s (1-θ^{(s)}_{τcl}) K^a_s S^a_s  -  (Σ_ζ Σ_i ζ ν^{ζ}_i).
        Then E_{τ_cl} = Σ_s (1-θ^{(s)}_{τcl}) K^a_s (S_cl - S^a_s)  (linear supply). :contentReference[oaicite:13]{index=13}
        """
        # Makers at τ_cl-1
        sum_K_m = 0.0
        sum_KS_m = 0.0
        for (Ki, Si) in state.supply_funcs.values():
            sum_K_m += float(Ki)
            sum_KS_m += float(Ki) * float(Si)

        # Agent all orders up to τ_cl-1, masked by θ_{τ_cl}
        sum_K_a = 0.0
        sum_KS_a = 0.0
        for s in range(0, self._h):
            mask = 1 - int(state.theta[s])
            if mask <= 0:
                continue
            Ks = float(self._agent_K[s])
            Ss = float(self._agent_S[s])
            sum_K_a += mask * Ks
            sum_KS_a += mask * Ks * Ss

        # Takers up to τ_cl-1
        taker_imbalance = 0.0
        taker_imbalance += +sum(float(v) for v in state.taker_volumes_plus)   # sellers
        taker_imbalance += -sum(float(v) for v in state.taker_volumes_minus)  # buyers

        den = sum_K_m + sum_K_a
        if den <= 1e-12:
            S_cl = float(state.H_cl)  # fallback
        else:
            S_cl = ((sum_KS_m + sum_KS_a) - taker_imbalance) / den

        # Executed volume for the agent at S_cl (linear supply evaluation)
        executed = 0.0
        for s in range(0, self._h):
            mask = 1 - int(state.theta[s])
            if mask <= 0:
                continue
            Ks = float(self._agent_K[s])
            Ss = float(self._agent_S[s])
            executed += mask * Ks * (S_cl - Ss)

        return float(S_cl), float(executed)

    # ---------------------------------------------------------------------
    # Convenience: expose pending cancel vector for env to apply into θ
    # ---------------------------------------------------------------------
    @property
    def pending_cancel(self) -> Optional[List[int]]:
        """Return the last c_t provided via set_agent_order (or None if not set)."""
        return getattr(self, "_pending_cancel", None)

    def clear_pending_cancel(self) -> None:
        """Clear the stored c_t after the environment applies it to θ_t."""
        if hasattr(self, "_pending_cancel"):
            delattr(self, "_pending_cancel")
