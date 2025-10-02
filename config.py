from dataclasses import dataclass
from typing import Optional

@dataclass
class EnvConfig:
    # Global horizon & grids
    tau_open: int                # τ_op
    tau_close: int               # τ_cl
    alpha: float                 # tick size α
    beta: float                  # quantization for Ka (auction)
    B: int                       # ±B tick bounds from mid
    V_max: int                   # max limit order size per step
    L_c: int                     # OB depth bound for CLOB
    L_a: int                     # max # auction makers
    L_takers: int                # bound on taker arrivals
    I_max: int                   # inventory cap I
    initial_mid: float           # initial Smid

    # Fake-data (emulator) params
    lambda_arrival: float        # Poisson rate for takers in CLOB
    v_market: int                # per-arrival volume in CLOB
    sigma_mid: float             # Smid Brownian vol
    pareto_vm: float             # Pareto vm (auction)
    pareto_gamma: float          # Pareto γ (auction)
    K_l: float                   # Ki lower
    K_u: float                   # Ki upper
    eps_contract: float          # ε for Lipschitz constraint in G
    M_side_grid: int             # max |Z| for Si = Smid + αZ
    p_new_mm: float              # Bernoulli prob Bt (new maker)
    p_cancel_mm: float           # Bernoulli prob Dt (cancel maker)
    p_new_taker: float           # Bernoulli prob J±_t (new taker)
    p_cancel_taker: float        # Bernoulli prob Gt (cancel taker)
    gamma_smooth: float          # γ smoothing for H_cl update (Alg. 1)

    # Penalty coefficients
    kappa: float                 # κ tick penalty factor
    q_wrong_side: float          # q wrong-side penalty (auction)
    d_cancel: float              # d cancellation unit cost
    lambda_term: float           # λ terminal penalty on inventory

@dataclass
class DQNConfig:
    obs_dim: int
    n_actions: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    lr: float = 1e-3
    gamma: float = 1.0  # finite horizon; we also time-index
    tau_target: float = 0.005
    dueling: bool = False
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    batch_size: int = 256
    buffer_size: int = 500_000
    train_start: int = 5_000
    train_freq: int = 1
    target_update_freq: int = 1_000
    seed: Optional[int] = 42

@dataclass
class TrainConfig:
    episodes: int
    log_interval: int = 10
    eval_every: int = 50
    eval_episodes: int = 10
    max_steps_per_episode: Optional[int] = None  # defaults to τ_cl+1
