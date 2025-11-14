import math
import itertools
import random
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Optional
import copy

from dataclasses import dataclass, field

# try:
#     plt.rcParams['text.usetex'] = True
# except Exception:
#     plt.rcParams['text.usetex'] = False

@dataclass
@dataclass
class EpisodeTracker:
    t: list = field(default_factory=list)
    phase: list = field(default_factory=list)  # 'C' or 'A'
    mid: list = field(default_factory=list)
    H_cl: list = field(default_factory=list)
    inv: list = field(default_factory=list)
    depth_ask: list = field(default_factory=list)
    depth_bid: list = field(default_factory=list)
    top_ask: list = field(default_factory=list)
    top_bid: list = field(default_factory=list)
    N_plus: list = field(default_factory=list)
    N_minus: list = field(default_factory=list)
    last_exec: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    cum_reward: list = field(default_factory=list)
    act_v: list = field(default_factory=list)        # CLOB actions
    act_delta: list = field(default_factory=list)
    act_K: list = field(default_factory=list)        # Auction actions
    act_S: list = field(default_factory=list)

def plot_episode(tr: EpisodeTracker, ep: int, tau_op: int, tau_cl: int, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    T = np.arange(len(tr.t))
    fig, axs = plt.subplots(3, 3, figsize=(12, 9))
    T = lambda arr: tr.t[:len(arr)]
    
    # 1) prices
    axs[0,0].plot(tr.t, tr.mid, label='$S_t^\mathrm{mid}$')
    axs[0,0].plot(tr.t, tr.H_cl, label='$H_t^\mathrm{cl}$', alpha=0.8)
    axs[0,0].axvline(tau_op, ls='--', lw=0.8, color='k')
    axs[0,0].set_title('$S_t^\mathrm{mid}$ vs $H_t^\mathrm{cl}$'); axs[0,0].legend()

    # 2) inventory & exec
    axs[0,1].plot(tr.t, tr.inv, label='$I_t$')
    axs[0,1].axvline(tau_op, ls='--', lw=0.8, color='k'); axs[0,1].legend()
    axs[0,1].set_title('Inventory')

    axs[0,2].plot(tr.t, tr.last_exec, label='$E_t$')
    axs[0,2].axvline(tau_op, ls='--', lw=0.8, color='k')
    axs[0,2].set_title('Executed (this step)')

    # 3) depths & top-of-book vols
    axs[1,0].plot(tr.t, tr.depth_ask, label='$L_t^+$')
    axs[1,0].plot(tr.t, tr.depth_bid, label='$L_t^-$')
    axs[1,0].axvline(tau_op, ls='--', lw=0.8, color='k'); axs[1,0].legend()
    axs[1,0].set_title('Depths')

    axs[1,1].plot(tr.t, tr.top_ask, label='$V_t^{+,1}$')
    axs[1,1].plot(tr.t, tr.top_bid, label='$V_t^{-,1}$')
    axs[1,1].axvline(tau_op, ls='--', lw=0.8, color='k'); axs[1,1].legend()
    axs[1,1].set_title('Top-of-book volumes')

    axs[1,2].plot(tr.t, tr.N_plus, label='$N_t^+$')
    axs[1,2].plot(tr.t, tr.N_minus, label='$N_t^-$')
    axs[1,2].axvline(tau_op, ls='--', lw=0.8, color='k'); axs[1,2].legend()
    axs[1,2].set_title('Auction MO counters')

    # 4) rewards
    axs[2,0].plot(tr.t, tr.reward); axs[2,0].set_title('$R_t$')
    axs[2,0].axvline(tau_op, ls='--', lw=0.8, color='k')

    axs[2,1].plot(tr.t, tr.cum_reward); axs[2,1].set_title('Cumulative reward')
    axs[2,1].axvline(tau_op, ls='--', lw=0.8, color='k')

    # 5) actions
    # show CLOB actions (v, Δ) and Auction (K, S) on same axis with NaNs to break lines
    axs[2,2].plot(T(tr.act_v), tr.act_v, label='$v_t$ (CLOB)')
    axs[2,2].plot(T(tr.act_delta), tr.act_delta, label='$\delta_t$ (CLOB)')
    axs[2,2].plot(T(tr.act_K), tr.act_K, label='$K_t^a$ (Auction)')
    axs[2,2].plot(T(tr.act_S), tr.act_S, label='$S_t^a$ (Auction)')
    axs[2,2].axvline(tau_op, ls='--', lw=0.8, color='k')
    axs[2,2].legend(); axs[2,2].set_title('Actions')

    for ax in axs.flat:
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Episode {ep} timeline", y=0.995)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.savefig("episode_{}.png".format(ep))
    plt.show()

class MarketEmulator:
    def __init__(self, tau_op=100, tau_cl=120, I=100, V=5000, L=10, Lc=8, La=5,
             lambda_param=0.005, kappa=0.1, q=1.0, d=1.0, gamma=0.5, 
             v_m=1000, pareto_gamma=2.0, poisson_rate=0.5, sigma_mid=1.0, seed=None,
             V_top_max=50000.0, lambda_decision=1.0, alpha=1.0, beta_a=2.0, beta_b=5.0, 
             depth_decay=0.6, price_band_ticks=5.0,
             L_max_auction=100):
        """
        Initialize the market emulator with given parameters.
        """
        self._seed: Optional[int] = None
        self.rng: np.random.Generator = np.random.default_rng()  # default
        if seed is not None:
            self.set_seed(seed)
        
        # Save parameters
        self.tau_op = tau_op
        self.tau_cl = tau_cl
        self.I_max = I
        self.V_max = V
        self.L_max_clob = L  # Max depth during CLOB phase (book depth)
        self.L_max = L_max_auction  # Max number of market orders during auction
        self.Lc = Lc
        self.La = La
        self.lambda_param = lambda_param
        self.kappa = kappa
        self.q = q
        self.d = d
        self.gamma = gamma
        self.v_m = v_m
        self.pareto_shape = pareto_gamma
        self.poisson_rate = poisson_rate
        self.sigma_mid = sigma_mid
        self.V_top_max = float(V_top_max)
        self.beta_a = float(beta_a)
        self.beta_b = float(beta_b)
        self.depth_decay = float(depth_decay)
        self.price_band_ticks = float(price_band_ticks)
        self.lambda_decision = lambda_decision
        self.alpha = float(alpha)

        # Running moments across decision times i for each price tick k
        self._mom_sum = defaultdict(float)     # ∑_{s≤i} vol_s^k
        self._mom_sum_sq = defaultdict(float)  # ∑_{s≤i} (vol_s^k)^2
        self._mom_count = 0    
        # Initialize state
        self.reset()
        
    def set_seed(self, seed: int) -> None:
        """Set env seed for both new Generator and legacy global RNG."""
        s = int(seed)
        self._seed = s
        # Generator-based RNG for any code that uses self.rng
        self.rng = np.random.default_rng(s)
        # Also seed legacy APIs in case the code uses np.random.*
        np.random.seed(s % (2**32 - 1))
        # If any Python 'random' calls are used, seed them too
        random.seed(s)

        
    def _refresh_order_book(self):
    # Fresh, independent Beta draws for ask/bid L1
        V1a = float(self.V_top_max * np.random.beta(self.beta_a, self.beta_b))
        V1b = float(self.V_top_max * np.random.beta(self.beta_a, self.beta_b))
        rho = self.depth_decay
        self.ask_volumes = [V1a * (rho ** j) for j in range(self.Lc)]
        self.bid_volumes = [V1b * (rho ** j) for j in range(self.Lc)]
        self.depth_ask = self.Lc
        self.depth_bid = self.Lc
        
    def _sample_mo_volume(self):
        # Pareto Type I: support [v_m, ∞), then truncated at V_max
        U = np.random.rand()
        vol = self.v_m / ((1.0 - U) ** (1.0 / self.pareto_shape))
        return float(min(vol, self.V_max))

    
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.set_seed(seed)
        """Reset the environment to the beginning of an episode."""
        # Phase and time
        self.phase = 'continuous'
        self.current_time = 0.0
        # Inventory and executed volume
        self.inventory = float(self.I_max)
        self.last_executed = 0.0
        # Track executed sell prices during the CLOB phase for the auction policy
        self.clob_sell_prices = []     # list of price levels at which sells occurred
        self.clob_sold_max_price = None

        # Mid-price and clearing price
        self.mid_price = 100.0  # arbitrary initial mid price
        self.H_cl = self.mid_price  # initialize hypothetical clearing price
        self._mom_sum.clear(); self._mom_sum_sq.clear(); self._mom_count = 0
        # Initialize order book volumes for continuous phase
        self._refresh_order_book() 
        # Market taker counters (cumulative number of arrivals) and volumes
        self.N_plus = 0   # number of sell MOs (arrived up to now)
        self.N_minus = 0  # number of buy MOs
        self.market_sell_volumes = [0.0] * self.L_max  # volumes of each sell MO
        self.market_buy_volumes = [0.0] * self.L_max   # volumes of each buy MO
        # Track agent's active order in continuous (price level and remaining volume)
        self.agent_active_order_cont = None
        # Auction phase order book (supply functions) and agent orders
        self.active_supply = []  # list of (K_i, S_i) for each exogenous supply function active
        self.agent_orders_K = [0.0] * (self.tau_cl + 1)
        self.agent_orders_S = [0.0] * (self.tau_cl + 1)
        self.agent_order_active = [False] * (self.tau_cl + 1)
        # Last continuous action time (for computing next decision time)
        self.last_action_time = 0.0
        self.current_time = 0.0
        self.next_decision_time = self.current_time + np.random.exponential(1.0 / self.lambda_decision)
        # Return initial state
        return self._get_state()
    
    def _get_state(self):
        """Assemble the current state vector components X_t^1,...,X_t^{17}."""
        t = self.current_time
        in_continuous = (self.phase == 'continuous')
        in_auction = (self.phase == 'auction')
        # X^1: Inventory
        X1 = self.inventory
        # X^2: Executed volume at terminal time (else 0)
        X2 = self.last_executed if t == self.tau_cl else 0.0
        # X^3: Hypothetical clearing price H_t^cl
        X3 = self.H_cl
        # X^4, X^5: Order book depths on ask and bid (only in continuous phase)
        X4 = self.depth_ask if in_continuous else 0
        X5 = self.depth_bid if in_continuous else 0
        # X^6: Number of active exogenous supply functions (only in auction)
        X6 = len(self.active_supply) if in_auction else 0
        # X^7, X^8: Cumulative number of market orders on each side (auction only)
        X7 = self.N_plus if in_auction else 0
        X8 = self.N_minus if in_auction else 0
        # X^9: Agent's cancellation status vector theta_t (length tau_cl+1)
        if in_auction:
            theta = []
            for s in range(self.tau_op, self.tau_cl):
                if self.agent_orders_K[s] != 0 or self.agent_orders_S[s] != 0:
                    canceled = 0 if self.agent_order_active[s] else 1
                else:
                    canceled = 0
                theta.append(canceled)
            theta.append(0)  # padding for index tau_cl
        else:
            theta = [0] * (self.tau_cl + 1)
        X9 = theta
        # X^10: Mid price
        X10 = self.mid_price
        # X^11, X^12: Volumes of each market order (auction only, up to L orders per side)
        if in_auction:
            X11 = [self.market_sell_volumes[i] if i < self.N_plus else 0.0 for i in range(self.L_max)]
            X12 = [self.market_buy_volumes[i]  if i < self.N_minus else 0.0 for i in range(self.L_max)]
        else:
            X11 = [0.0] * self.L_max
            X12 = [0.0] * self.L_max
        # X^13, X^14: Order book volumes at each level on ask and bid (continuous phase only)
        if in_continuous:
            X13 = []
            for j in range(self.Lc):
                vol = self.ask_volumes[j] if j < len(self.ask_volumes) else 0.0
                if j+1 > self.depth_ask: 
                    vol = 0.0  # no volume beyond depth
                X13.append(vol)
            X14 = []
            for j in range(self.Lc):
                vol = self.bid_volumes[j] if j < len(self.bid_volumes) else 0.0
                if j+1 > self.depth_bid:
                    vol = 0.0
                X14.append(vol)
        else:
            X13 = [0.0] * self.Lc
            X14 = [0.0] * self.Lc
        # X^15: Values of each active exogenous supply function on the price grid (alpha * [-B, B])
        B = 10  # price grid half-width in ticks
        k_mid = round(self.mid_price / self.alpha)
        price_grid = [self.alpha * (k_mid + i) for i in range(-B, B+1)]
        X15 = []
        if in_auction:
            for idx in range(self.La):
                if idx < len(self.active_supply):
                    K_i, S_i = self.active_supply[idx]
                    values = [K_i * (p - S_i) for p in price_grid]
                    X15.append(values)
                else:
                    X15.append([0.0] * len(price_grid))
        else:
            X15 = [[0.0] * len(price_grid) for _ in range(self.La)]
        # X^16, X^17: Agent's past order parameters S_s^a and K_s^a (for s < t)
        X16 = []
        X17 = []
        for s in range(int(self.tau_cl) + 1):
            if self.tau_op <= s < t:
                X16.append(self.agent_orders_S[s])
                X17.append(self.agent_orders_K[s])
            else:
                X16.append(0.0)
                X17.append(0.0)
        # Return state
        return {
            'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5,
            'X6': X6, 'X7': X7, 'X8': X8, 'X9': X9, 'X10': X10,
            'X11': X11, 'X12': X12, 'X13': X13, 'X14': X14,
            'X15': X15, 'X16': X16, 'X17': X17,
            'time': self.current_time
        }
        
    def _update_hyp_clearing_price_from_book(self):
        """
        Implements Algorithm 1 at a continuous decision time:
        - Build O_i from current standing (unexecuted) volumes on both sides.
        - Update running moments ê_i^k and σ̂_i^k.
        - Compute K̂_i^k = (2 ê_i^k - σ̂_i^k / ê_i^k) / tick_size (when ê_i^k>0).
        - Solve ∑_k K̂_i^k (tick_size k - p) = 0  ⇒  p = (∑_k K̂_i^k tick_size k) / (∑_k K̂_i^k).
        - Smooth H_cl towards p with parameter γ.
        """
        eps = 1e-12
        tick_size = self.alpha

        # ---- Build O_i: outstanding aggregated volume per absolute tick k
        # Map our book (relative to mid) to integer ticks k ∈ ℤ
        k0 = int(round(self.mid_price / tick_size))  # reference tick near the current mid
        vol_by_k = {}

        # ask levels: prices tick_size(k0 + j)
        for j in range(min(self.Lc, len(self.ask_volumes))):
            v = float(self.ask_volumes[j])
            if v > eps:
                k = k0 + j
                vol_by_k[k] = vol_by_k.get(k, 0.0) + v

    # bid levels: prices tick_size(k0 - j)
        for j in range(min(self.Lc, len(self.bid_volumes))):
            v = float(self.bid_volumes[j])
            if v > eps:
                k = k0 - j
                vol_by_k[k] = vol_by_k.get(k, 0.0) + v
                
        if self.agent_active_order_cont and self.agent_active_order_cont['volume'] > 1e-12:
            j = int(self.agent_active_order_cont['level'])
            k = k0 + j  # same mapping you use for ask levels
            vol_by_k[k] = vol_by_k.get(k, 0.0) + float(self.agent_active_order_cont['volume'])

        # ---- Update running moments over i
        self._mom_count += 1
        for k, v in vol_by_k.items():
            self._mom_sum[k]    += v
            self._mom_sum_sq[k] += v * v

        # ---- Compute K̂_i^k for all k with positive mean
        num = 0.0  # ∑ K̂_i^k tick_size k
        den = 0.0  # ∑ K̂_i^k
        i = self._mom_count
        for k, s in self._mom_sum.items():
            e_hat = s / i                         # ê_i^k
            if e_hat <= eps:
                continue
            sig_hat = self._mom_sum_sq[k] / i     # σ̂_i^k
            K_hat = max(0.0, (2.0 * e_hat - sig_hat / max(e_hat, eps)) / tick_size)  # K̂_i^k
            if K_hat > 0.0:
                num += K_hat * (tick_size * k)
                den += K_hat

        # ---- Solve for p and smooth H_cl
        if den > 0.0:
            tilde_S = num / den
            self.H_cl = self.H_cl + self.gamma * (tilde_S - self.H_cl)
        # else: leave H_cl unchanged (degenerate empty O_i, which we don't expect)
    
    def step(self, action):
        """
        Advance the environment by one decision step given the agent's action.
        Returns (next_state, reward, done).
        """
        # Helper function: positive part
        def f(x): return x if x > 0 else 0
        
        # -----------------------------
        # Continuous (CLOB) phase step
        # -----------------------------
        if self.phase == 'continuous':
            # Expect action = (v, delta) for volume and price offset (in ticks, delta>=0)
            v, delta = action
            v = float(v); delta = float(delta)
            # Ensure action is admissible
            if v > self.inventory: 
                v = self.inventory
            v = max(0.0, min(v, self.V_max))
            # If an old order is still active (leftover), cancel it (no cost in continuous)
            self.agent_active_order_cont = None

            # Clamp level into book and snap price to tick
            j_agent = int(max(0, min(self.Lc - 1, math.floor(delta))))
            k_mid = math.floor(self.mid_price / self.alpha)
            agent_price = self.alpha * (k_mid + j_agent)

            # Place agent's order in the OB
            if v > 0 and j_agent < self.Lc:
                self.agent_active_order_cont = {'level': j_agent, 'price': agent_price, 'volume': v}
            
            # Simulate random market order arrivals until next agent action time
            last_time = self.current_time
            # Next discrete decision boundary (end of second or end of continuous)
            next_allowed_time = min(math.floor(last_time) + 1, self.tau_op - 1)
            target_time = float(next_allowed_time)

            executed_vol = 0.0   # total executed volume of agent's order in this interval

            # Initialize next arrival times (independent Poissons)
            # Use NumPy exponentials for speed: scale=1/lambda
            next_buy_time = last_time + np.random.exponential(scale=1.0 / self.poisson_rate)
            next_sell_time = last_time + np.random.exponential(scale=1.0 / self.poisson_rate)

            # Define event processing for a buy market order (consumes ask side)
            def process_buy_order(volume):
                nonlocal executed_vol
                remain = float(volume)

                # 1) Consume ask-side volumes up to agent's level
                upto = min(j_agent, self.Lc)
                for j in range(upto):
                    if remain <= 0:
                        break
                    lvl = self.ask_volumes[j]
                    if lvl <= 0:
                        continue
                    take = min(remain, lvl)
                    self.ask_volumes[j] -= take
                    remain -= take

                # 2) Agent priority at j_agent
                if remain > 0 and self.agent_active_order_cont and j_agent < self.Lc:
                    vol_agent = self.agent_active_order_cont['volume']
                    take = min(remain, vol_agent)
                    if take > 0:
                        executed_vol += take
                        self.inventory -= take
                        self.agent_active_order_cont['volume'] = vol_agent - take
                        remain -= take
                        if self.agent_active_order_cont['volume'] <= 1e-9:
                            self.agent_active_order_cont = None

                # 3) Exogenous volume at j_agent and deeper
                if remain > 0:
                    if j_agent < self.Lc:
                        lvl = self.ask_volumes[j_agent]
                        take = min(remain, lvl)
                        self.ask_volumes[j_agent] -= take
                        remain -= take
                    j = j_agent + 1
                    while remain > 0 and j < self.Lc:
                        lvl = self.ask_volumes[j]
                        take = min(remain, lvl)
                        self.ask_volumes[j] -= take
                        remain -= take
                        j += 1
            
            # Define event processing for a sell market order (consumes bid side)
            def process_sell_order(volume):
                remain = float(volume)
                for j in range(self.Lc):
                    if remain <= 0:
                        break
                    lvl = self.bid_volumes[j]
                    if lvl <= 0:
                        continue
                    take = min(remain, lvl)
                    self.bid_volumes[j] -= take
                    remain -= take
            
            last_time = self.current_time
            # enforce: at least one buy and one sell arrival since last decision
            tau_plus  = next_buy_time
            tau_minus = next_sell_time
            tau_i = max(tau_plus, tau_minus)

            # optional deterministic grid: t_i = floor(last_time)+1
            t_i = math.floor(last_time) + 1.0

            target_time = min(max(t_i, tau_i), self.tau_op - 1.0)

            # Simulate events up to the decision boundary
            while min(next_buy_time, next_sell_time) <= target_time:
                if next_buy_time <= next_sell_time:
                    current_time = next_buy_time
                    process_buy_order(volume=self._sample_mo_volume())
                    next_buy_time = current_time + np.random.exponential(scale=1.0 / self.poisson_rate)
                else:
                    current_time = next_sell_time
                    process_sell_order(volume=self._sample_mo_volume())
                    next_sell_time = current_time + np.random.exponential(scale=1.0 / self.poisson_rate)
                    

            # Decide at the boundary (always advance at least to target_time)
            next_decision_time = target_time

            # Update order book depths after processing events
            self.depth_ask = next((j+1 for j,v in enumerate(self.ask_volumes) if v <= 1e-6), self.Lc)
            self.depth_bid = next((j+1 for j,v in enumerate(self.bid_volumes) if v <= 1e-6), self.Lc)
            
            # Algorithm 1: update H_cl from outstanding LOs
            self._update_hyp_clearing_price_from_book()
                    

            # Compute reward for this step
            S_submit = agent_price  # agent's limit price
            E_t = executed_vol       # executed volume of agent's order
            reward = S_submit * E_t * f(1 - self.kappa * f(self.H_cl - S_submit))
            
            # Record executed sell prices for the CLOB phase
            # Assumes E_t > 0.0 means a sell (inventory decreased) at price S_submit.
            if E_t > 0.0:
                self.clob_sell_prices.append(float(S_submit))
                if (self.clob_sold_max_price is None) or (float(S_submit) > self.clob_sold_max_price):
                    self.clob_sold_max_price = float(S_submit)

            # Update state variables
            self.last_executed = E_t
            self.inventory = float(np.clip(self.inventory, -self.I_max, self.I_max))

            # Advance current time to the next decision time
            dt = max(0.0, target_time - last_time)
            self.current_time = float(target_time)
            self.last_action_time = self.current_time
            
            self._refresh_order_book()

            # Remove any remaining agent order (it will be replaced or canceled at next action)
            self.agent_active_order_cont = None

            # Update mid-price via Brownian motion for the time interval
            if dt > 0:
                self.mid_price += float(np.random.normal(0.0, self.sigma_mid * math.sqrt(dt)))
                
            self.next_decision_time = self.current_time + np.random.exponential(1.0 / self.lambda_decision)

            # Check for transition to auction phase
            done = False
            if self.current_time >= self.tau_op - 1:
                # Transition to auction
                self.phase = 'auction'
                self.current_time = float(self.tau_op)
                self.N_plus = 0; self.N_minus = 0
                self.market_sell_volumes = [0.0] * self.L_max
                self.market_buy_volumes = [0.0] * self.L_max
                self.active_supply = [] 

            next_state = self._get_state()
            return next_state, reward, done
        
        # -----------------------------
        # Auction phase step
        # -----------------------------
        elif self.phase == 'auction':
            # Expect action = (K_t^a, S_t^a, c_t)
            K_a, S_a, c_t = action
            K_a = float(K_a); S_a = float(S_a)
            t = int(self.current_time)
            done = False

            # 1. Process exogenous arrivals/cancellations at time t
            # (a) Exogenous limit (supply) orders
            if np.random.rand() < 0.3 and len(self.active_supply) < self.La:
                K_new = np.random.uniform(0.1, 2.0)
                S_new = self.mid_price + np.random.uniform(-self.price_band_ticks, self.price_band_ticks) * self.alpha
                self.active_supply.append((float(K_new), float(S_new)))
            if np.random.rand() < 0.2 and self.active_supply:
                idx = int(np.random.randint(0, len(self.active_supply)))
                self.active_supply.pop(idx)
            # (b) Exogenous market orders (Pareto capped by V_max)
            if np.random.rand() < 0.3:
                U = np.random.rand()
                vol = self.v_m / ((1.0 - U) ** (1.0 / self.pareto_shape))
                vol = float(min(vol, self.V_max))
                if self.N_plus < self.L_max:
                    self.market_sell_volumes[self.N_plus] = vol
                self.N_plus = min(self.N_plus + 1, self.L_max)
            if np.random.rand() < 0.3:
                U = np.random.rand()
                vol = self.v_m / ((1.0 - U) ** (1.0 / self.pareto_shape))
                vol = float(min(vol, self.V_max))
                if self.N_minus < self.L_max:
                    self.market_buy_volumes[self.N_minus] = vol
                self.N_minus = min(self.N_minus + 1, self.L_max)
            # Random cancellation of a pending market order
            if np.random.rand() < 0.1:
                if self.N_plus > 0 and np.random.rand() < 0.5:
                    idx = int(np.random.randint(0, self.N_plus))
                    self.market_sell_volumes[idx] = 0.0
                if self.N_minus > 0 and np.random.rand() < 0.5:
                    idx = int(np.random.randint(0, self.N_minus))
                    self.market_buy_volumes[idx] = 0.0

            # 2. Apply agent's cancellations c_t
            if isinstance(c_t, torch.Tensor):
                c_t = c_t.tolist()
            for s in range(self.tau_op, t):
                if s < len(c_t) and c_t[s] == 1 and self.agent_order_active[s]:
                    self.agent_order_active[s] = False

            # 3. Agent submits new order (K_t^a, S_t^a)
            if K_a > 0:
                self.agent_orders_K[t] = K_a
                self.agent_orders_S[t] = S_a
                self.agent_order_active[t] = True

            # 4. Update hypothetical clearing price H_t^cl at time t
            total_slope = 0.0
            total_intercept_term = 0.0
            for (K_i, S_i) in self.active_supply:
                total_slope += K_i
                total_intercept_term += K_i * S_i
            for s in range(self.tau_op, t+1):
                if self.agent_order_active[s]:
                    total_slope += self.agent_orders_K[s]
                    total_intercept_term += self.agent_orders_K[s] * self.agent_orders_S[s]
            net_order_volume = 0.0
            if self.N_plus > 0:
                net_order_volume += float(sum(self.market_sell_volumes[:self.N_plus]))
            if self.N_minus > 0:
                net_order_volume -= float(sum(self.market_buy_volumes[:self.N_minus]))
            if total_slope > 0:
                self.H_cl = (total_intercept_term - net_order_volume) / total_slope
            else:
                self.H_cl = self.mid_price

            # 5. Immediate reward at time t
            reward = 0.0
            reward += K_a * (self.H_cl - S_a)                          # market-making revenue
            reward -= self.q * f(- K_a * (self.H_cl - S_a))            # wrong-side penalty
            cancel_count = 0
            if isinstance(c_t, (list, tuple)):
                cancel_count = sum(1 for s in range(self.tau_op, t) if s < len(c_t) and c_t[s] == 1)
            reward -= self.d * cancel_count

            # 6. Advance time to next auction tick
            self.last_executed = 0.0
            self.current_time = float(t + 1)

            # 7. Final clearing
            if self.current_time == float(self.tau_cl):
                total_slope = 0.0
                total_intercept_term = 0.0
                for (K_i, S_i) in self.active_supply:
                    total_slope += K_i
                    total_intercept_term += K_i * S_i
                for s in range(self.tau_op, self.tau_cl):
                    if self.agent_order_active[s]:
                        total_slope += self.agent_orders_K[s]
                        total_intercept_term += self.agent_orders_K[s] * self.agent_orders_S[s]
                net_order_volume = 0.0
                if self.N_plus > 0:
                    net_order_volume += float(sum(self.market_sell_volumes[:self.N_plus]))
                if self.N_minus > 0:
                    net_order_volume -= float(sum(self.market_buy_volumes[:self.N_minus]))
                if total_slope > 0:
                    clearing_price = (total_intercept_term - net_order_volume) / total_slope
                else:
                    clearing_price = self.H_cl

                # Executed "volume" per your supply function convention
                executed_final = 0.0
                for s in range(self.tau_op, self.tau_cl):
                    if self.agent_order_active[s]:
                        executed_final += self.agent_orders_K[s] * (clearing_price - self.agent_orders_S[s])

                # Update inventory and clamp to bounds
                I_final = self.inventory - executed_final
                I_final = float(np.clip(I_final, -self.I_max, self.I_max))
                self.inventory = I_final
                self.last_executed = executed_final

                # Terminal reward components
                terminal_reward = executed_final - self.lambda_param * (abs(I_final) ** 2)
                wrong_side_sum = 0.0
                for s in range(self.tau_op, self.tau_cl):
                    if self.agent_order_active[s]:
                        wrong_side_sum += f(- self.agent_orders_K[s] * (clearing_price - self.agent_orders_S[s]))
                terminal_reward -= self.q * wrong_side_sum
                reward += terminal_reward
                done = True

            next_state = self._get_state()
            return next_state, reward, done
        
        # After terminal
        else:
            return self._get_state(), 0.0, True
        
# ---------------------
# GLFT liquidation benchmark (sell-only on CLOB; no trading in auction)
# ---------------------
from dataclasses import dataclass
import math
import numpy as np

try:
    from scipy.linalg import expm  # optional fast path for matrix exponential
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# put near your GLFT runner
from contextlib import contextmanager

@contextmanager
def benchmark_penalty_override(env, *, disable_wrong_side=True, disable_cancel=True):
    """Temporarily disable penalties that are irrelevant to the GLFT benchmark."""
    old_q, old_d = float(env.q), float(env.d)
    if disable_wrong_side:
        env.q = 0.0        # kills wrong-side penalties in auction ticks and at terminal
    if disable_cancel:
        env.d = 0.0        # kills per-tick cancellation costs
    try:
        yield
    finally:
        env.q, env.d = old_q, old_d

@dataclass
class GLFTParams:
    A: float              # base arrival intensity (per unit time)
    k: float              # intensity decay per *currency* unit (NOT per tick)
    gamma: float          # CARA risk aversion
    sigma: float          # mid-price vol (per time unit of env, same units as S)
    alpha_tick: float     # tick size (currency per tick)
    I_max: int            # inventory cap Q (set to I0 for strict seller)
    T: int                # last CLOB index; T = tau_op - 1 (auction starts at tau_op)
    dt: float = 1.0       # env time step

    # --- GLFT scalars (currency-units formulas) ---
    @property
    def alpha_GLFT(self) -> float:
        # α = k * γ * σ^2 / 2  (currency-units; NO tick factor here)
        if self.gamma == 0.0:
            return 0.0
        return self.k * self.gamma * (self.sigma ** 2) / 2.0

    @property
    def eta_GLFT(self) -> float:
        # η = A * (1 + γ/k)^{-(1 + k/γ)} with γ->0 limit = A/e
        if self.gamma == 0.0:
            return self.A / math.e
        r = 1.0 + self.gamma / self.k
        return self.A * (r ** (-(1.0 + self.k / self.gamma)))

    # --- helpful currency<->tick conversions ---
    @property
    def k_per_tick(self) -> float:
        # if δ is expressed in ticks, λ(δ_ticks) = A * exp(-(k * α_tick) * δ_ticks)
        return self.k * self.alpha_tick

    @property
    def inv_k_per_tick(self) -> float:
        return 1.0 / self.k_per_tick

    @property
    def const_ticks(self) -> float:
        # δ*_ticks second term:
        #   currency form: (1/γ) ln(1 + γ/k)
        #   ticks form    : (1/(γ α_tick)) ln(1 + γ/k)
        if self.gamma == 0.0:
            # limit (1/γ) ln(1 + γ/k) -> 1/k, then / α_tick
            return 1.0 / (self.k * self.alpha_tick)
        return (1.0 / (self.gamma * self.alpha_tick)) * math.log(1.0 + self.gamma / self.k)


class GLFTBenchmark:
    def __init__(self, params: GLFTParams, clob_actions, q_name_in_state="inv"):
        """
        clob_actions: iterable of (volume_in_shares, delta_in_ticks)
                      Only (v>0, any delta) are used to place sells.
                      If you also have a (0, anything) "no-trade" action, we'll prefer it when q=0.
        """
        self.p = params
        self.clob_actions = list(clob_actions)
        self.q_name = q_name_in_state

        # precompute outputs
        self.v = None               # shape (T+1, I_max+1), time-major, v[t,q]
        self.delta_star = None      # shape (T+1, I_max+1), in ticks

        # index allowed (v>0) CLOB actions and build delta grid
        self.allowed_clob = [(i, a) for i, a in enumerate(self.clob_actions) if a[0] > 0]
        self.delta_grid = np.array([a[1] for _, a in self.allowed_clob], dtype=float)

    # ---------------------
    # Core GLFT: build M and compute v(t)
    # ---------------------
    def _build_M_sell_only(self):
        """
        M is (I_max+1) x (I_max+1) with ascending q order: [v_0, v_1, ..., v_I]^T.
        From v_q'(t) = α q^2 v_q(t) - η v_{q-1}(t), q>=1; and v_0'(t) = 0.
          -> diag: M[q,q]   = α * q^2  (M[0,0] = 0)
          -> subdiag: M[q,q-1] = -η for q>=1
        NOTE: do NOT add +η to the diagonal.
        """
        I = self.p.I_max
        M = np.zeros((I + 1, I + 1), dtype=float)

        # diagonal
        q = np.arange(I + 1, dtype=float)
        M[np.arange(I + 1), np.arange(I + 1)] = self.p.alpha_GLFT * (q ** 2)
        M[0, 0] = 0.0  # explicit

        # subdiagonal -η
        eta = self.p.eta_GLFT
        for qq in range(1, I + 1):
            M[qq, qq - 1] = -eta
        return M

    def _solve_v_timegrid(self, M):
        """
        Compute v[t, q] = [exp( -M * (T - t) * dt ) * 1]_q, for t=0..T, q=0..I.
        - SciPy path: exact matrix exponential
        - Fallback path:
            * If α≈0 (risk-neutral limit), exploit the closed form for strictly
              lower-bidiagonal M (nilpotent): v_q(t) = sum_{n=0}^q (τ η)^n / n!
            * Else, use a stable backward Euler stepping on the ODE system for robustness.
        """
        I, T, dt = self.p.I_max, self.p.T, self.p.dt
        one = np.ones(I + 1, dtype=float)
        alpha = self.p.alpha_GLFT
        eta = self.p.eta_GLFT

        if _HAVE_SCIPY:
            E = expm(-M * self.p.dt)
            v = np.empty((T + 1, I + 1))
            v[T] = 1.0
            for t in range(T - 1, -1, -1):
                v[t] = E @ v[t + 1]
            return v

        # --- risk-neutral closed form (α=0) ---
        if abs(alpha) < 1e-14:
            v = np.empty((T + 1, I + 1), dtype=float)
            for t in range(T + 1):
                tau = (T - t) * dt
                z = tau * eta
                # v_q(t) = sum_{n=0}^q z^n / n!
                # compute cumulatively for numerical stability
                term = 1.0  # n=0
                s = [term]
                for q_idx in range(1, I + 1):
                    term *= z / q_idx
                    s.append(s[-1] + term)
                v[t] = np.array(s, dtype=float)
            return v

        # --- generic robust fallback: backward Euler time stepping (stable for stiff lower-triangular systems) ---
        # integrate w'(τ) = -M w, with w(0)=1, out to τ = (T - t)*dt for each t
        # We solve once from τ=0 up to τ=T*dt, recording at integer τ_k = k*dt.
        K = T  # integer steps of size dt
        w = one.copy()  # at τ=0
        v = np.empty((T + 1, I + 1), dtype=float)
        v[T] = w.copy()  # τ=0 -> t=T
        # Pre-factor (I + dt M)^{-1} once (M is lower-bidiagonal -> easy solve)
        A = np.eye(I + 1) + dt * M
        # we’ll solve A w_{k+1} = w_k via forward-substitution (A is lower-bidiagonal)
        for k in range(K):
            # forward substitution for lower-bidiagonal system
            w_new = np.empty_like(w)
            # row 0: A00 * w0 = w_old0
            w_new[0] = w[0] / A[0, 0]
            for i in range(1, I + 1):
                w_new[i] = (w[i] - A[i, i - 1] * w_new[i - 1]) / A[i, i]
            w = w_new
            v[T - (k + 1)] = w.copy()
        return v

    def precompute(self):
        """
        Exact (or stable) precompute of v and δ* from the matrix exponential / ODE.
        """
        M = self._build_M_sell_only()
        v = self._solve_v_timegrid(M)

        # shapes & terminal condition
        assert v.shape == (self.p.T + 1, self.p.I_max + 1)
        if not np.allclose(v[-1], 1.0, rtol=1e-10, atol=1e-12):
            raise RuntimeError("v(T,·) != 1; check M construction / solver.")

        self.v = v
        self.delta_star = self._compute_deltas(v)
        assert self.delta_star.shape == (self.p.T + 1, self.p.I_max + 1)

    # ---------------------
    # Optimal quotes δ*_ticks
    # ---------------------
    def _compute_deltas(self, v):
        """
        Currency formula:  δ*_cur = (1/k) log(v_q/v_{q-1}) + (1/γ) log(1 + γ/k)  (γ->0 gives +1/k)
        Ticks formula   :  δ*_ticks = (1/(k α_tick)) log(v_q/v_{q-1}) + (1/(γ α_tick)) log(1 + γ/k)
                           and in γ->0: δ*_ticks = (1/(k α_tick)) [ log(v_q/v_{q-1}) + 1 ].
        Defined for q>=1; for q=0 we set +inf (no meaningful ask if no inventory).
        """
        I, T = self.p.I_max, self.p.T
        delt = np.full((T + 1, I + 1), np.inf, dtype=float)  # time-major
        inv_k_tick = self.p.inv_k_per_tick
        const_ticks = self.p.const_ticks

        eps = 1e-16
        for t in range(T + 1):
            for q in range(1, I + 1):
                ratio = v[t, q] / max(v[t, q - 1], eps)
                ratio = max(ratio, eps)
                delt[t, q] = inv_k_tick * math.log(ratio) + const_ticks
            # q=0 -> +inf (won't be used because we won't trade without inventory)
            delt[t, 0] = math.inf
        return delt

    # ---------------------
    # Wiring into a discrete-action env
    # ---------------------
    def _state_get_inventory(self, s, env=None):
        CANDIDATE_KEYS = (self.q_name, 'inv', 'inventory', 'I', 'q', 'qty', 'position')

        def try_extract(obj):
            if obj is None:
                return None
            if isinstance(obj, dict):
                for k in CANDIDATE_KEYS:
                    if k in obj:
                        return obj[k]
            for k in CANDIDATE_KEYS:
                if hasattr(obj, k):
                    return getattr(obj, k)
            return None

        q = try_extract(s)
        if q is None and isinstance(s, (tuple, list)):
            for elem in s:
                q = try_extract(elem)
                if q is not None:
                    break
        if q is None and env is not None:
            q = try_extract(getattr(env, 'state', None))
        if q is None and env is not None:
            q = try_extract(env)
        if q is None:
            raise AttributeError(f"Cannot find inventory; looked for {CANDIDATE_KEYS} in state/env.")

        try:
            q = int(q)
        except Exception:
            if isinstance(q, np.ndarray) and q.shape == ():
                q = int(q.item())
            else:
                q = int(float(q))
        return max(0, min(q, self.p.I_max))

    def choose_clob_action_index(self, s, t_idx, env=None):
        """
        Map continuous δ*_ticks(t,q) to nearest discrete (v, δ_ticks) in clob_actions.
        Among actions with v>0, clamp by inventory and prefer the largest feasible v
        at the chosen δ to accelerate liquidation.
        """
        q = self._state_get_inventory(s, env=env)
        if q <= 0:
            # Prefer an explicit no-trade action if present
            for i, a in enumerate(self.clob_actions):
                if a[0] == 0:
                    return i
            # else: pick minimal volume with the largest delta to minimize fills
            if self.allowed_clob:
                i, _ = max(self.allowed_clob, key=lambda p: p[1][1])  # largest δ
                return i
            return 0  # last resort

        t_idx = int(np.clip(int(t_idx), 0, self.p.T))
        delta_star = self.delta_star[t_idx, min(q, self.p.I_max)]
        if not np.isfinite(delta_star) or len(self.allowed_clob) == 0:
            # very conservative: largest δ among allowed
            idx, _ = max(self.allowed_clob, key=lambda p: p[1][1])
            return idx

        # nearest δ in the discrete grid
        j = int(np.argmin(np.abs(self.delta_grid - delta_star)))
        idx, (v, d) = self.allowed_clob[j]

        # try to pick the largest volume at that δ not exceeding q
        candidates = [(i, a) for i, a in self.allowed_clob if a[1] == d and a[0] <= q]
        if candidates:
            idx, _ = max(candidates, key=lambda p: p[1][0])
        return idx

    @staticmethod
    def find_auction_noop_index(auction_actions):
        for i, a in enumerate(auction_actions):
            # (0,0,0) or obj with flag
            if a == (0, 0, 0) or getattr(a, "is_noop", False):
                return i
        return 0


# ---------------------
# Replay episodes with GLFT liquidation (CLOB only), no trading in auction
# ---------------------
def run_glft_benchmark_episodes(env, glft: GLFTBenchmark, episode_seeds, auction_actions=None, ignore_wrong_side=False, ignore_cancel=False):
    """
    Replays len(episode_seeds) episodes with identical randomness,
    choosing GLFT actions in the CLOB and a no-op in the auction.
    Returns: list of episode returns.
    """
    import random
    try:
        import torch
        _HAVE_TORCH = True
    except Exception:
        _HAVE_TORCH = False

    bm_returns = []
    with benchmark_penalty_override(env,
                                    disable_wrong_side=ignore_wrong_side,
                                    disable_cancel=ignore_cancel):

        # --- helpers ---
        def _noop_clob(env):
            return (0.0, 0.0)  # (v, delta)

        def _noop_auction(env):
            n = max(1, int(getattr(env, "auction_horizon", 1)))
            S_mid = float(getattr(env, "mid_price", 0.0))
            return (0.0, S_mid, [0.0] * n)  # (K, S, c_t)

        def _normalize_step_out(out):
            if not isinstance(out, tuple):
                raise TypeError(f"env step returned non-tuple: {type(out)}")
            n = len(out)
            if n == 5:
                s, r, terminated, truncated, info = out
                return s, r, bool(terminated or truncated), info
            if n == 4:
                s, r, done, info = out
                return s, r, bool(done), info
            if n == 3:
                s, r, done = out
                return s, r, bool(done), {}
            raise ValueError(f"Unsupported step return length {n}. Expected 3, 4, or 5.")

        def _step4(env, preferred_attr, action):
            fn = getattr(env, preferred_attr, None)
            out = fn(action) if callable(fn) else env.step(action)
            return _normalize_step_out(out)

        # Resolve auction no-op from action list if provided (optional)
        AUCTION_NOOP_IDX = None
        if auction_actions is not None:
            AUCTION_NOOP_IDX = GLFTBenchmark.find_auction_noop_index(auction_actions)

        for seed in episode_seeds:
            s = env.reset(seed=seed)
            done = False
            cum_r = 0.0
            t_idx = 0

            while not done:
                r_accum = 0.0  # collect all rewards for the iteration
                phase_before = getattr(env, "phase", "C")

                # -------- CLOB PHASE --------
                if phase_before in ("continuous", "C", "clob"):
                    # choose discrete GLFT action index
                    a_idx = glft.choose_clob_action_index(s, t_idx, env=env)
                    if hasattr(env, "step_clob"):
                        s, r, done, info = _step4(env, "step_clob", a_idx)   # index-based
                    else:
                        v, delta = glft.clob_actions[a_idx]                   # (volume, delta_in_ticks)
                        s, r, done, info = _step4(env, "step", (float(v), float(delta)))
                    r_accum += float(r)

                    # If we just flipped into AUCTION, send one opening-auction limit order per the policy
                    phase_after = getattr(env, "phase", None)
                    if phase_after in ("auction", "A"):
                        q_left = float(getattr(env, "inventory", 0.0))
                        if q_left > 0.0 and not done:
                            # We require that some sells actually occurred during CLOB to define max/mean
                            prices = getattr(env, "clob_sell_prices", None)
                            if prices and len(prices) > 0:
                                pmax = max(prices)
                                # "mean price" = arithmetic mean of executed sell prices in CLOB.
                                # If your definition of "mean price" differs, replace the next line accordingly.
                                pmean = sum(prices) / len(prices)

                                S_star = 0.5 * (pmax + pmean)          # price at auction opening
                                n = max(1, int(getattr(env, "auction_horizon", 1)))
                                K_big = 10.0 * q_left                  # encodes full remaining volume at that price
                                auct_action = (K_big, S_star, [0.0] * n)

                                s, r, done, info_liq = _step4(env, "step_auction", auct_action) \
                                    if hasattr(env, "step_auction") else _step4(env, "step", auct_action)
                                r_accum += float(r)

                                if isinstance(info, dict):
                                    info.update({
                                        "auction_opening_order": True,
                                        "I_tau_op": q_left,
                                        "p_max_CLOB": pmax,
                                        "p_mean_CLOB": pmean,
                                        "S_star": S_star
                                    })


                # -------- AUCTION PHASE --------
                elif phase_before in ("auction", "A"):
                    # no trading in the auction (true benchmark), but still advance the env
                    if auction_actions is not None and AUCTION_NOOP_IDX is not None:
                        noop = auction_actions[AUCTION_NOOP_IDX]  # (K, S, c_t)
                    else:
                        noop = _noop_auction(env)                 # (K, S, c_t)
                    s, r, done, info = _step4(env, "step_auction", noop) \
                        if hasattr(env, "step_auction") else _step4(env, "step", noop)
                    r_accum += float(r)

                # -------- UNKNOWN PHASE: advance safely --------
                else:
                    s, r, done, info = _step4(env, "step_clob", _noop_clob(env)) \
                        if hasattr(env, "step_clob") else _step4(env, "step", _noop_clob(env))
                    r_accum += float(r)

                cum_r += r_accum
                t_idx += 1

            bm_returns.append(cum_r)

    return bm_returns

# ---------------------
# Config
# ---------------------

SEED = 42
DEVICE = torch.device("cpu")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Create env (stable-ish defaults)
env = MarketEmulator(
    tau_op=120, tau_cl=150,     # 100 discrete “seconds” of CLOB, 50 ticks of auction
    I=100,                     # inventory cap
    V=30,                      # hard cap per order (both agent & exo MOs)
    L=12, Lc=12, La=12,
    L_max_auction=100,        # book depth & auction supply capacity

    # costs / penalties (tune to taste)
    lambda_param=0.2,         # leftover inventory penalty weight for I^2
    kappa=0.1,                  # concavity in revenue term (your f(·))
    q=0.5,                      # wrong-side penalty weight
    d=0.05,                     # cancel cost
    gamma=0.95,                  # smoothing for H_cl update

    # MO size distribution (used in both continuous & auction)
    v_m=2.0,                   # Pareto scale
    pareto_gamma=2.5,           # Pareto tail index (2–3 = realistic-ish)
    poisson_rate=0.5,           # MOs per side per unit time (≈3 total/sec)
    sigma_mid=0.10,             # mid-price vol per sqrt(time); ~1 tick over 100 steps
    alpha=1.0,                  # tick size
    seed=None,

    # book shape / L1 stats
    V_top_max=15.0,            # cap for L1 volume draws
    beta_a=2.0, beta_b=5.0,     # skew toward smaller L1
    depth_decay=0.5,            # geometric decay per level (ρ)
    price_band_ticks=5.0        # auction supply centers within ±5 ticks
)

# Actions: smaller steps
# V_CHOICES = np.array([0, 0.1, 0.2, 0.4, 0.7, 1.0]) * env.V_max
V_CHOICES = np.array([0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]) * env.V_max
DELTA_CHOICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
CLOB_ACTIONS = [(0.0, 0)] + [(v, d) for v in V_CHOICES[1:] for d in DELTA_CHOICES]

STEPS_AUCT = max(1, env.tau_cl - env.tau_op)
K_MAX = 8.0 * env.I_max / STEPS_AUCT # enough slope to clear over the horizon
K_CHOICES = [0.0] + list(np.linspace(1.0, K_MAX, 10)) # Encourage being on the sell side at the close
S_OFFSETS = [-12, -10, -8, -6, -4, -2, 0, 2, 4]
AUCT_ACTIONS = [(K, off, cancel) 
                for K in K_CHOICES
                for off in S_OFFSETS
                for cancel in (0, 1)]
# Training tweaks
EPISODES = 10000
LR = 4e-3          # a bit smaller
GAMMA = 0.998     # slightly longer credit

# ---------------------
# GLFT params (tune/calibrate as you like)
# ---------------------

GLFT_A       = 0.5        # base hit intensity (arbitrary scale)
GLFT_k       = 0.5        # per-currency decay; *effective* slope is k*alpha in ticks
GLFT_gamma   = 0.0        # CARA risk aversion
GLFT_sigma   = float(getattr(env, "sigma_mid", 0.2))        # per-step mid-price vol used in benchmark (match your env)
GLFT_T       = int(getattr(env, "tau_op", 100)) - 1    # CLOB horizon; falls back to 100
GLFT_I_MAX   = int(getattr(env, "I_max", 1000))           # inventory cap I
GLFT_ALPHA   = float(getattr(env, "alpha", 1.0))    # tick size

glft_params = GLFTParams(
    A=GLFT_A, k=GLFT_k, gamma=GLFT_gamma, sigma=GLFT_sigma,
    alpha_tick=GLFT_ALPHA, I_max=GLFT_I_MAX, T=GLFT_T, dt=1.0
)

# Build the benchmark (uses your global CLOB_ACTIONS)
glft = GLFTBenchmark(glft_params, CLOB_ACTIONS, q_name_in_state="inv")  # change name if your state uses a different attr
glft.precompute()

# ---------------------
# Feature extraction
# ---------------------
def feat_clob(s, env):
    I_max, Lc, Vmax = env.I_max, env.Lc, env.V_max
    t_norm = (s['time'] / max(1.0, (env.tau_op - 1))) if s['time'] <= env.tau_op - 1 else 1.0
    top_ask = (s['X13'][0] / Vmax) if len(s['X13']) > 0 and Vmax > 0 else 0.0
    top_bid = (s['X14'][0] / Vmax) if len(s['X14']) > 0 and Vmax > 0 else 0.0
    x_list = [
        s['X1'] / max(1.0, I_max),
        s['X3'],
        s['X10'],
        s['X4'] / max(1, Lc),
        s['X5'] / max(1, Lc),
        top_ask,
        top_bid,
        t_norm,
    ]
    return torch.tensor(x_list, dtype=torch.float32, device=DEVICE)

def feat_auction(s, env):
    I_max = env.I_max
    t_norm = (s['time'] - env.tau_op) / max(1.0, (env.tau_cl - env.tau_op))
    x_list = [
        s['X1'] / max(1.0, I_max),
        s['X3'],
        s['X10'],
        s['X6'] / max(1, env.La),
        s['X7'] / max(1, env.L_max),
        s['X8'] / max(1, env.L_max),
        t_norm,
    ]
    return torch.tensor(x_list, dtype=torch.float32, device=DEVICE)

# ---------------------
# Tiny DQNs
# ---------------------
class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 50), nn.ReLU(),
            nn.Linear(50, 50), nn.ReLU(),
            nn.Linear(50, out_dim)
        )
    def forward(self, x):
        return self.net(x)

clob_net = DQN(in_dim=8, out_dim=len(CLOB_ACTIONS)).to(DEVICE)
auct_net = DQN(in_dim=7, out_dim=len(AUCT_ACTIONS)).to(DEVICE)

clob_target = DQN(in_dim=8, out_dim=len(CLOB_ACTIONS)).to(DEVICE)
auct_target = DQN(in_dim=7, out_dim=len(AUCT_ACTIONS)).to(DEVICE)
clob_target.load_state_dict(clob_net.state_dict()); clob_target.eval()
auct_target.load_state_dict(auct_net.state_dict()); auct_target.eval()

print("Saving initial (untrained) network states...")
initial_clob_net_state = copy.deepcopy(clob_net.state_dict())
initial_auct_net_state = copy.deepcopy(auct_net.state_dict())

def soft_update(target, source, tau=0.01):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1 - tau).add_(sp.data, alpha=tau)


opt_clob = optim.Adam(clob_net.parameters(), lr=LR)
opt_auct = optim.Adam(auct_net.parameters(), lr=LR)
REWARD_SCALE = 1e-3  # scale down rewards by 1000
mse = nn.SmoothL1Loss()  # Huber

# ε-greedy schedule
#def epsilon_by_episode(ep, start=1.0, end=0.02, total=EPISODES):
#    if total <= 1: return end
#    frac = ep / (total - 1)
#    return start + (end - start) * frac

def epsilon_by_episode(ep, start=1.0, end=0.02, total=EPISODES, warmup=100):
    if total <= 1: return end
    if ep < warmup:
        # Linear warmup from start
        return start
    # Exponential decay after warmup
    decay_rate = -np.log(end / start) / (total - warmup)
    return start * np.exp(-decay_rate * (ep - warmup))

# ---------------------
# Storage for plots 
# ---------------------
all_mid_paths = []
all_step_rewards = []
all_cum_rewards = []
final_inventories = []

def run_eval_episode(env, clob_net, auct_net, seed=None):
    s = env.reset(seed=seed)
    done = False
    total_r = 0.0
    while not done:
        if env.phase == 'continuous':
            x = feat_clob(s, env).unsqueeze(0)
            with torch.no_grad():
                a_idx = int(torch.argmax(clob_net(x)[0]).item())
            v, delta = CLOB_ACTIONS[a_idx]
            v = min(v, s['X1'])
            s, r, done = env.step((v, delta))
        else:
            x = feat_auction(s, env).unsqueeze(0)
            with torch.no_grad():
                a_idx = int(torch.argmax(auct_net(x)[0]).item())
            K, off, cancel = AUCT_ACTIONS[a_idx]
            S = s['X10'] + off * env.alpha
            if cancel:
                # cancel all prior auction-time posts
                c_vec = [0]*(env.tau_cl+1)
                t = int(s['time'])
                for s_cancel in range(env.tau_op, t):
                    c_vec[s_cancel] = 1
            else:
                c_vec = [0]*(env.tau_cl+1)
            s, r, done = env.step((K, S, c_vec))
        total_r += r
    return total_r

# ---------------------
# Loss tracking (per episode)
# ---------------------
clob_loss_per_ep = []
auct_loss_per_ep = []

# ---------------------
# Training loop
# ---------------------
PLOT_EVERY_EPISODE = True   # ← toggle
SAVE_EP_FIGS = False        # set a path like f"plots/ep_{ep}.png" if you want files
PLOT_EVERY_N = 2500          # 0/None disables the cadence filter

# Collectors for regret analysis
DQN_RETURNS = []

# --- evaluation config ---
EVAL_INTERVAL = 10
N_EVAL_EPISODES = 32
# Using precomputed EVAL_SEEDS from RNG snapshots above
# EVAL_SEEDS = EPISODE_SEEDS[:N_EVAL_EPISODES]
eval_returns = []  # collect eval curve you’ll plot

MASTER_SEED = 42
EVAL_MASTER_SEED = 120

_rng_train = np.random.default_rng(MASTER_SEED)
EPISODE_SEEDS = _rng_train.integers(0, 2**32 - 1, size=EPISODES, dtype=np.uint32).tolist()

_rng_eval = np.random.default_rng(EVAL_MASTER_SEED)
EVAL_SEEDS = _rng_eval.integers(0, 2**32 - 1, size=N_EVAL_EPISODES, dtype=np.uint32).tolist()

FINAL_EVAL_MASTER_SEED = 200
N_FINAL_EVAL_EPISODES = 100

_rng_final_eval = np.random.default_rng(FINAL_EVAL_MASTER_SEED)
FINAL_EVAL_SEEDS = _rng_final_eval.integers(0, 2**32 - 1, size=N_FINAL_EVAL_EPISODES, dtype=np.uint32).tolist()

print(f"\nGenerated {N_FINAL_EVAL_EPISODES} fresh evaluation seeds")

for ep in range(EPISODES):
    seed = int(EPISODE_SEEDS[ep])
    eps = epsilon_by_episode(ep)
    s = env.reset(seed=seed)
    
    # --- INITIAL LOGGING so the very first point shows I_0 (e.g., 100) ---
    tracker = EpisodeTracker()
    mid_track = []
    r_track = []
    cum_track = []
    cum_r = 0.0
    done = False

    # initial logs (from the state returned by reset)
    mid_track.append(s['X10'])                         # mid
    r_track.append(0.0)
    cum_track.append(0.0)
    tracker.t.append(s['time'])
    tracker.phase.append('C')
    tracker.mid.append(s['X10'])
    tracker.H_cl.append(s['X3'])
    tracker.inv.append(s['X1'])                        # inventory at t=0 (should be 100)
    tracker.depth_ask.append(s['X4'])
    tracker.depth_bid.append(s['X5'])
    tracker.top_ask.append(s['X13'][0] if len(s['X13']) else 0.0)
    tracker.top_bid.append(s['X14'][0] if len(s['X14']) else 0.0)
    tracker.N_plus.append(s['X7'])
    tracker.N_minus.append(s['X8'])
    tracker.last_exec.append(0.0)
    tracker.reward.append(0.0)
    tracker.cum_reward.append(0.0)
    # --- end initial logging ---

    # (removed duplicate tracker reinitialization)
    
    # --- loss accumulators for this episode ---
    clob_loss_sum, clob_updates = 0.0, 0
    auct_loss_sum, auct_updates = 0.0, 0

    while not done:
        phase_before = env.phase  # capture BEFORE choosing action
        if phase_before == 'continuous':
            x = feat_clob(s, env).unsqueeze(0)
            with torch.no_grad():
                q_vals = clob_net(x)[0]
            a_idx = random.randrange(len(CLOB_ACTIONS)) if random.random() < eps else int(torch.argmax(q_vals).item())
            v, delta = CLOB_ACTIONS[a_idx]
            v = min(v, s['X1'])

            s2, r, done = env.step((v, delta))

            if env.phase == 'continuous' and not done:
                x2 = feat_clob(s2, env).unsqueeze(0)
                with torch.no_grad():
                    q_next = clob_target(x2).max(dim=1)[0]
            else:
                if not done:
                    x2 = feat_auction(s2, env).unsqueeze(0)
                    with torch.no_grad():
                        q_next = auct_target(x2).max(dim=1)[0]
                else:
                    q_next = torch.tensor([0.0], device=DEVICE)


            target = torch.tensor([r * REWARD_SCALE], device=DEVICE) + GAMMA * q_next
            q_sa = clob_net(x)[0, a_idx].unsqueeze(0)
            loss = mse(q_sa, target.detach())
            opt_clob.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(clob_net.parameters(), 1.0)
            opt_clob.step()
            soft_update(clob_target, clob_net, tau=0.01)
            
            # loss bookkeeping (continuous net)
            clob_loss_sum += float(loss.item())
            clob_updates += 1

            # --- logging for tracker (post-step state s2) ---
            tracker.act_v.append(v)
            tracker.act_delta.append(delta)
            tracker.act_K.append(float('nan'))
            tracker.act_S.append(float('nan'))

        else:
            x = feat_auction(s, env).unsqueeze(0)
            with torch.no_grad():
                q_vals = auct_net(x)[0]
            a_idx = random.randrange(len(AUCT_ACTIONS)) if random.random() < eps else int(torch.argmax(q_vals).item())
            K, off, cancel = AUCT_ACTIONS[a_idx]
            S = s['X10'] + off * env.alpha
            if cancel:
                # cancel all prior auction-time posts
                c_vec = [0]*(env.tau_cl+1)
                t = int(s['time'])
                for s_cancel in range(env.tau_op, t):
                    c_vec[s_cancel] = 1
            else:
                c_vec = [0]*(env.tau_cl+1)

            s2, r, done = env.step((K, S, c_vec))

            if env.phase == 'auction' and not done:
                x2 = feat_auction(s2, env).unsqueeze(0)
                with torch.no_grad():
                    q_next = auct_target(x2).max(dim=1)[0]
            else:
                q_next = torch.tensor([0.0], device=DEVICE)


            target = torch.tensor([r * REWARD_SCALE], device=DEVICE) + GAMMA * q_next
            q_sa = auct_net(x)[0, a_idx].unsqueeze(0)
            loss = mse(q_sa, target.detach())
            opt_auct.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(auct_net.parameters(), 1.0)
            opt_auct.step()
            soft_update(auct_target, auct_net, tau=0.01)
            
            # loss bookkeeping (auction net)
            auct_loss_sum += float(loss.item())
            auct_updates += 1

            # --- logging for tracker ---
            tracker.act_v.append(float('nan'))
            tracker.act_delta.append(float('nan'))
            tracker.act_K.append(K)
            tracker.act_S.append(S)

        # common logging (use s2)
        cum_r += r
        r_track.append(r)
        cum_track.append(cum_r)
        mid_track.append(s2['X10'])

        tracker.t.append(s2['time'])
        tracker.phase.append('A' if env.phase == 'auction' else 'C')  # phase AFTER step
        tracker.mid.append(s2['X10'])
        tracker.H_cl.append(s2['X3'])
        tracker.inv.append(s2['X1'])
        tracker.depth_ask.append(s2['X4'])
        tracker.depth_bid.append(s2['X5'])
        tracker.top_ask.append(s2['X13'][0] if len(s2['X13']) else 0.0)
        tracker.top_bid.append(s2['X14'][0] if len(s2['X14']) else 0.0)
        tracker.N_plus.append(s2['X7'])
        tracker.N_minus.append(s2['X8'])
        tracker.last_exec.append(env.last_executed)
        tracker.reward.append(r)
        tracker.cum_reward.append(cum_r)    

        s = s2

    DQN_RETURNS.append(cum_r)
    
    # --- evaluation on fixed seeds (greedy; no learning) ---
    if (ep + 1) % EVAL_INTERVAL == 0:
        eval_r = float(np.mean([run_eval_episode(env, clob_net, auct_net, seed=int(s)) 
                        for s in EVAL_SEEDS]))
        eval_returns.append(eval_r)
    else:
        eval_returns.append(eval_returns[-1] if len(eval_returns) else np.nan)
    
    all_mid_paths.append(mid_track)
    all_step_rewards.append(r_track)
    all_cum_rewards.append(cum_track)
    final_inventories.append(env.inventory)

    do_plot = PLOT_EVERY_EPISODE and (not PLOT_EVERY_N or (ep+1) % PLOT_EVERY_N == 0)
    # --- end-of-episode loss means ---
    clob_loss_per_ep.append(clob_loss_sum / max(1, clob_updates))
    auct_loss_per_ep.append(auct_loss_sum / max(1, auct_updates))
    if do_plot:
        save_path = None
        if SAVE_EP_FIGS:
            import os; os.makedirs("plots", exist_ok=True)
            save_path = f"plots/episode_{ep+1}.png"
        plot_episode(tracker, ep+1, env.tau_op, env.tau_cl, save_path=save_path)

# ---------------------
# Plots
# ---------------------
# plt.figure(figsize=(7,4))
# for ep, path in enumerate(all_mid_paths, 1):
#     plt.plot(path, label=f"Ep {ep}")
# plt.title("Mid-price trajectories (decision steps)")
# plt.xlabel("Decision step"); plt.ylabel("Mid price")
# plt.legend(ncol=2, fontsize=8); plt.tight_layout(); plt.show()

# plt.figure(figsize=(7,4))
# for ep, rpath in enumerate(all_step_rewards, 1):
#     plt.plot(rpath, label=f"Ep {ep}")
# plt.title("Step rewards per episode")
# plt.xlabel("Decision step"); plt.ylabel("Reward")
# plt.legend(ncol=2, fontsize=8); plt.tight_layout(); plt.show()

# plt.figure(figsize=(7,4))
# for ep, cpath in enumerate(all_cum_rewards, 1):
#     plt.plot(cpath, label=f"Ep {ep}")
# plt.title("Cumulative reward per episode")
# plt.xlabel("Decision step"); plt.ylabel("Cumulative reward")
# plt.legend(ncol=2, fontsize=8); plt.tight_layout(); plt.show()

plt.figure(figsize=(6,4))
n = len(final_inventories)
plt.bar(range(1, n+1), final_inventories)
plt.axhline(0, color='k', lw=0.7)
plt.title("Final inventory by episode")
plt.xlabel("Episode"); plt.ylabel("Inventory at $\\tau^\mathrm{cl}$")
plt.savefig("final_inventories.png")
plt.tight_layout(); plt.show()

# print("Final inventories:", final_inventories)
# print("Total cumulative rewards:", [round(c[-1], 2) for c in all_cum_rewards])

# ---------------------
# Loss vs episodes
# ---------------------
plt.figure(figsize=(7,4))
n = len(clob_loss_per_ep)
plt.plot(range(1, n+1), clob_loss_per_ep, marker='o', label='CLOB DQN loss (mean/ep)')
plt.plot(range(1, n+1), auct_loss_per_ep, marker='o', label='Auction DQN loss (mean/ep)')
plt.xlabel("Episode")
plt.ylabel("MSE loss")
plt.title("DQN training loss per episode")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("dqn_training_loss.png")
plt.show()

print("Mean CLOB losses per episode:", [round(v, 6) for v in clob_loss_per_ep])
print("Mean Auction losses per episode:", [round(v, 6) for v in auct_loss_per_ep])

# ---------------------
# Plot evaluation returns vs GLFT benchmark
# ---------------------
# Compute GLFT benchmark returns at evaluation points
glft_eval_returns = []
for i in range(len(eval_returns)):
    if (i + 1) % EVAL_INTERVAL == 0:
        # Get the episode indices for this evaluation point
        ep_end = i + 1
        ep_start = max(0, ep_end - N_EVAL_EPISODES)
        eval_ep_seeds = EPISODE_SEEDS[ep_start:ep_end]
        
        # Run GLFT benchmark on these episodes
        bm_ret = run_glft_benchmark_episodes(
            env, glft, eval_ep_seeds, 
            auction_actions=None,
            ignore_wrong_side=False,
            ignore_cancel=False
        )
        glft_eval_returns.append(np.mean(bm_ret))
    else:
        # Carry forward previous value
        glft_eval_returns.append(glft_eval_returns[-1] if glft_eval_returns else np.nan)
        


plt.figure(figsize=(7, 4))
plt.plot(range(1, len(eval_returns) + 1), eval_returns, marker='o', label='DQN (greedy eval)', alpha=0.8)
plt.plot(range(1, len(glft_eval_returns) + 1), glft_eval_returns, marker='s', label='GLFT benchmark', alpha=0.8, linestyle='--')
plt.axhline(0, color='k', lw=0.7, linestyle=':')
plt.title("Evaluation returns: DQN vs GLFT benchmark")
plt.xlabel("Episode")
plt.ylabel("Mean return (greedy policy)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("eval_returns_vs_benchmark.png")
plt.show()

# ---------------------
# Evaluate GLFT on the same episodes, then compute & plot pseudo-regret
# ---------------------
# If you have a global AUCTION_ACTIONS, pass it so we can find a no-op
# --- right before calling the benchmark ---
N = min(len(DQN_RETURNS), len(EPISODE_SEEDS))
bm_returns = run_glft_benchmark_episodes(env, glft, EPISODE_SEEDS[:N], auction_actions=None)
DQN_RETURNS = DQN_RETURNS[:N]
assert len(bm_returns) == len(DQN_RETURNS) == N

# compute regret
pseudo_regret = [br - dr for br, dr in zip(bm_returns, DQN_RETURNS)]
cum_pseudo_regret = np.cumsum(pseudo_regret)

# Plot
plt.figure(figsize=(7.0, 4.0))
plt.plot(cum_pseudo_regret, label="Cumulative pseudo-regret (GLFT − DQN)")
plt.axhline(0.0, linestyle="--")
plt.xlabel("Episode")
plt.ylabel("Cumulative pseudo-regret")
plt.title("Regret analysis vs GLFT liquidation benchmark")
plt.legend()
plt.tight_layout()
plt.savefig("dqn_vs_glft_regret.png")
plt.show()

# (Optional) Also inspect episode-wise returns
plt.figure(figsize=(7.0, 4.0))
plt.plot(DQN_RETURNS, label="DQN return", alpha=0.8)
plt.plot(bm_returns, label="GLFT benchmark return", alpha=0.8)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Episode returns: DQN vs GLFT")
plt.legend()
plt.tight_layout()
plt.savefig("dqn_vs_glft_returns.png")
plt.show()

# =========================
# Trailing-window mean of returns (DQN vs GLFT)
# =========================

# Align lengths
N = min(len(DQN_RETURNS), len(bm_returns))
dqn_ret = np.asarray(DQN_RETURNS[:N], dtype=float)
bm_ret  = np.asarray(bm_returns[:N], dtype=float)

# Choose a window (feel free to tweak)
WINDOW = max(10, min(400, N // 10))   # e.g., ~10% of run, bounded [10,100]

def trailing_mean(x, w):
    if N < w:
        # not enough points: fallback to prefix means
        return np.cumsum(x) / (np.arange(1, len(x)+1))
    # simple moving average via cumulative sums (O(N))
    cs = np.cumsum(np.insert(x, 0, 0.0))
    sma = (cs[w:] - cs[:-w]) / float(w)
    # pad the first w-1 points with the best available shorter averages
    prefix = np.cumsum(x[:w-1]) / np.arange(1, w)
    return np.concatenate([prefix, sma])

dqn_avg = trailing_mean(dqn_ret, WINDOW)
bm_avg  = trailing_mean(bm_ret,  WINDOW)

plt.figure(figsize=(9, 4.5))
plt.plot(range(1, N + 1), dqn_avg, label=f"DQN {WINDOW}-episode avg", linewidth=2)
plt.plot(range(1, N + 1), bm_avg,  label=f"GLFT {WINDOW}-episode avg", linestyle="--", linewidth=2)

# Horizontal lines for the *last* window average only
plt.axhline(dqn_avg[-1], linestyle=":", linewidth=1,
            label=f"DQN last {WINDOW} avg = {dqn_avg[-1]:.2f}")
plt.axhline(bm_avg[-1],  linestyle=":", linewidth=1,
            label=f"GLFT last {WINDOW} avg = {bm_avg[-1]:.2f}")

plt.title(f"Trailing Mean of Episode Returns (window={WINDOW})")
plt.xlabel("Episode")
plt.ylabel(f"Mean Return over last {WINDOW} episodes")
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dqn_vs_glft_trailing_mean_returns.png")
plt.show()

def evaluate_policy(env, clob_net, auct_net, eval_seeds, policy_name="Policy"):
    """
    Evaluate a policy on given seeds with detailed statistics.
    Returns: dict with returns, inventory stats, and per-phase rewards
    """
    print(f"\nEvaluating {policy_name} on {len(eval_seeds)} episodes...")
    
    returns = []
    final_inventories = []
    clob_rewards = []
    auction_rewards = []
    
    for i, seed in enumerate(eval_seeds):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(eval_seeds)}")
        
        s = env.reset(seed=int(seed))
        done = False
        total_r = 0.0
        clob_r = 0.0
        auction_r = 0.0
        
        while not done:
            if env.phase == 'continuous':
                x = feat_clob(s, env).unsqueeze(0)
                with torch.no_grad():
                    a_idx = int(torch.argmax(clob_net(x)[0]).item())
                v, delta = CLOB_ACTIONS[a_idx]
                v = min(v, s['X1'])
                s, r, done = env.step((v, delta))
                clob_r += r
            else:  # auction
                x = feat_auction(s, env).unsqueeze(0)
                with torch.no_grad():
                    a_idx = int(torch.argmax(auct_net(x)[0]).item())
                K, off, cancel = AUCT_ACTIONS[a_idx]
                S = s['X10'] + off * env.alpha
                if cancel:
                    c_vec = [0]*(env.tau_cl+1)
                    t = int(s['time'])
                    for s_cancel in range(env.tau_op, t):
                        c_vec[s_cancel] = 1
                else:
                    c_vec = [0]*(env.tau_cl+1)
                s, r, done = env.step((K, S, c_vec))
                auction_r += r
            
            total_r += r
        
        returns.append(total_r)
        final_inventories.append(env.inventory)
        clob_rewards.append(clob_r)
        auction_rewards.append(auction_r)
    
    return {
        'returns': np.array(returns),
        'inventories': np.array(final_inventories),
        'clob_rewards': np.array(clob_rewards),
        'auction_rewards': np.array(auction_rewards),
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'median_return': np.median(returns),
        'mean_inventory': np.mean(final_inventories),
        'mean_clob_reward': np.mean(clob_rewards),
        'mean_auction_reward': np.mean(auction_rewards)
    }

# ---------------------
# 4. Evaluate GLFT benchmark
# ---------------------
print("\n" + "="*70)
print("POST-TRAINING EVALUATION")
print("="*70)

print("\n[1/3] Evaluating GLFT Benchmark...")
glft_results = run_glft_benchmark_episodes(
    env, glft, FINAL_EVAL_SEEDS, 
    auction_actions=None,
    ignore_wrong_side=False,
    ignore_cancel=False
)

glft_stats = {
    'returns': np.array(glft_results),
    'mean_return': np.mean(glft_results),
    'std_return': np.std(glft_results),
    'median_return': np.median(glft_results)
}

# ---------------------
# 5. Evaluate final (trained) DQN
# ---------------------
print("\n[2/3] Evaluating Final (Trained) DQN...")
final_dqn_stats = evaluate_policy(env, clob_net, auct_net, FINAL_EVAL_SEEDS, "Final DQN")

# ---------------------
# 6. Evaluate initial (untrained) DQN
# ---------------------
print("\n[3/3] Evaluating Initial (Untrained) DQN...")

# Create temporary networks and load initial states
initial_clob_net = DQN(in_dim=8, out_dim=len(CLOB_ACTIONS)).to(DEVICE)
initial_auct_net = DQN(in_dim=7, out_dim=len(AUCT_ACTIONS)).to(DEVICE)
initial_clob_net.load_state_dict(initial_clob_net_state)
initial_auct_net.load_state_dict(initial_auct_net_state)
initial_clob_net.eval()
initial_auct_net.eval()

initial_dqn_stats = evaluate_policy(env, initial_clob_net, initial_auct_net, 
                                     FINAL_EVAL_SEEDS, "Initial DQN")

# ---------------------
# 7. Print comparison table
# ---------------------
print("\n" + "="*70)
print("EVALUATION RESULTS SUMMARY")
print("="*70)

print(f"\n{'Metric':<30} {'Initial DQN':<15} {'Final DQN':<15} {'GLFT':<15}")
print("-" * 70)
print(f"{'Mean Return':<30} {initial_dqn_stats['mean_return']:>14.1f} {final_dqn_stats['mean_return']:>14.1f} {glft_stats['mean_return']:>14.1f}")
print(f"{'Std Return':<30} {initial_dqn_stats['std_return']:>14.1f} {final_dqn_stats['std_return']:>14.1f} {glft_stats['std_return']:>14.1f}")
print(f"{'Median Return':<30} {initial_dqn_stats['median_return']:>14.1f} {final_dqn_stats['median_return']:>14.1f} {glft_stats['median_return']:>14.1f}")
print(f"{'Mean Final Inventory':<30} {initial_dqn_stats['mean_inventory']:>14.2f} {final_dqn_stats['mean_inventory']:>14.2f} {'N/A':<15}")
print(f"{'Mean CLOB Reward':<30} {initial_dqn_stats['mean_clob_reward']:>14.1f} {final_dqn_stats['mean_clob_reward']:>14.1f} {'N/A':<15}")
print(f"{'Mean Auction Reward':<30} {initial_dqn_stats['mean_auction_reward']:>14.1f} {final_dqn_stats['mean_auction_reward']:>14.1f} {'N/A':<15}")

print("\n" + "-" * 70)
print("RELATIVE IMPROVEMENTS")
print("-" * 70)

improvement_vs_initial = 100 * (final_dqn_stats['mean_return'] / initial_dqn_stats['mean_return'] - 1)
improvement_vs_glft = 100 * (final_dqn_stats['mean_return'] / glft_stats['mean_return'] - 1)

print(f"Final DQN vs Initial DQN:     {improvement_vs_initial:>+6.1f}%")
print(f"Final DQN vs GLFT:            {improvement_vs_glft:>+6.1f}%")
print(f"GLFT vs Initial DQN:          {100*(glft_stats['mean_return']/initial_dqn_stats['mean_return']-1):>+6.1f}%")

# Statistical significance (t-test)
from scipy import stats

t_stat_final_vs_glft, p_val_final_vs_glft = stats.ttest_ind(
    final_dqn_stats['returns'], glft_stats['returns']
)
t_stat_final_vs_initial, p_val_final_vs_initial = stats.ttest_ind(
    final_dqn_stats['returns'], initial_dqn_stats['returns']
)

print("\n" + "-" * 70)
print("STATISTICAL SIGNIFICANCE (Two-sample t-test)")
print("-" * 70)
print(f"Final DQN vs GLFT:     t={t_stat_final_vs_glft:>6.2f}, p={p_val_final_vs_glft:.4f} {'***' if p_val_final_vs_glft < 0.001 else '**' if p_val_final_vs_glft < 0.01 else '*' if p_val_final_vs_glft < 0.05 else 'ns'}")
print(f"Final DQN vs Initial:  t={t_stat_final_vs_initial:>6.2f}, p={p_val_final_vs_initial:.4f} {'***' if p_val_final_vs_initial < 0.001 else '**' if p_val_final_vs_initial < 0.01 else '*' if p_val_final_vs_initial < 0.05 else 'ns'}")

# ---------------------
# 8. Visualization
# ---------------------

# Plot 1: Return distributions (violin plots)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Box plots with individual points
ax = axes[0]
data_to_plot = [initial_dqn_stats['returns'], final_dqn_stats['returns'], glft_stats['returns']]
positions = [1, 2, 3]
labels = ['Initial\nDQN', 'Final\nDQN', 'GLFT']

bp = ax.boxplot(data_to_plot, positions=positions, widths=0.5, patch_artist=True,
                showmeans=True, meanline=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                meanprops=dict(color='green', linewidth=2, linestyle='--'))

# Add scattered points with jitter
for i, (data, pos) in enumerate(zip(data_to_plot, positions)):
    y = data
    x = np.random.normal(pos, 0.04, size=len(y))
    ax.plot(x, y, 'o', alpha=0.3, markersize=4, color='navy')

ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_ylabel('Episode Return', fontsize=12)
ax.set_title('Return Distributions', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='black', linewidth=0.8, linestyle=':')

# Right: Histogram comparison
ax = axes[1]
bins = np.linspace(
    min(initial_dqn_stats['returns'].min(), final_dqn_stats['returns'].min(), glft_stats['returns'].min()),
    max(initial_dqn_stats['returns'].max(), final_dqn_stats['returns'].max(), glft_stats['returns'].max()),
    30
)

ax.hist(initial_dqn_stats['returns'], bins=bins, alpha=0.5, label='Initial DQN', density=True, color='red')
ax.hist(final_dqn_stats['returns'], bins=bins, alpha=0.5, label='Final DQN', density=True, color='blue')
ax.hist(glft_stats['returns'], bins=bins, alpha=0.5, label='GLFT', density=True, color='orange')

ax.axvline(initial_dqn_stats['mean_return'], color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(final_dqn_stats['mean_return'], color='blue', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(glft_stats['mean_return'], color='orange', linestyle='--', linewidth=2, alpha=0.7)

ax.set_xlabel('Episode Return', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Return Distribution Density', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("final_evaluation_distributions.png", dpi=150)
plt.show()

# Plot 2: Bar chart with error bars
fig, ax = plt.subplots(figsize=(10, 6))

policies = ['Initial DQN', 'Final DQN', 'GLFT']
means = [initial_dqn_stats['mean_return'], final_dqn_stats['mean_return'], glft_stats['mean_return']]
stds = [initial_dqn_stats['std_return'], final_dqn_stats['std_return'], glft_stats['std_return']]
colors = ['red', 'blue', 'orange']

x_pos = np.arange(len(policies))
bars = ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, color=colors, 
               edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 100,
            f'{mean:.0f}\n±{std:.0f}',
            ha='center', va='bottom', fontsize=11)

ax.set_xticks(x_pos)
ax.set_xticklabels(policies, fontsize=12)
ax.set_ylabel('Mean Episode Return', fontsize=13)
ax.set_title('Post-Training Policy Comparison', fontsize=15)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='black', linewidth=1, linestyle=':')

plt.tight_layout()
plt.savefig("final_evaluation_comparison.png", dpi=150)
plt.show()

# Plot 3: Phase-wise reward breakdown (Final DQN only)
fig, ax = plt.subplots(figsize=(10, 6))

phase_labels = ['CLOB Phase', 'Auction Phase', 'Total']
final_values = [
    final_dqn_stats['mean_clob_reward'],
    final_dqn_stats['mean_auction_reward'],
    final_dqn_stats['mean_return']
]
initial_values = [
    initial_dqn_stats['mean_clob_reward'],
    initial_dqn_stats['mean_auction_reward'],
    initial_dqn_stats['mean_return']
]

x_pos = np.arange(len(phase_labels))
width = 0.35

bars1 = ax.bar(x_pos - width/2, initial_values, width, label='Initial DQN', alpha=0.7, color='red')
bars2 = ax.bar(x_pos + width/2, final_values, width, label='Final DQN', alpha=0.7, color='blue')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

ax.set_xticks(x_pos)
ax.set_xticklabels(phase_labels, fontsize=12)
ax.set_ylabel('Mean Reward', fontsize=13)
ax.set_title('Phase-wise Reward Breakdown', fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='black', linewidth=1, linestyle=':')

plt.tight_layout()
plt.savefig("final_evaluation_phase_breakdown.png", dpi=150)
plt.show()

# Plot 4: Final inventory distribution (DQN only)
fig, ax = plt.subplots(figsize=(10, 6))

bins_inv = np.linspace(
    min(initial_dqn_stats['inventories'].min(), final_dqn_stats['inventories'].min()),
    max(initial_dqn_stats['inventories'].max(), final_dqn_stats['inventories'].max()),
    30
)

ax.hist(initial_dqn_stats['inventories'], bins=bins_inv, alpha=0.6, 
        label=f"Initial DQN (mean={initial_dqn_stats['mean_inventory']:.2f})", 
        color='red', edgecolor='black')
ax.hist(final_dqn_stats['inventories'], bins=bins_inv, alpha=0.6, 
        label=f"Final DQN (mean={final_dqn_stats['mean_inventory']:.2f})", 
        color='blue', edgecolor='black')

ax.axvline(initial_dqn_stats['mean_inventory'], color='red', linestyle='--', linewidth=2)
ax.axvline(final_dqn_stats['mean_inventory'], color='blue', linestyle='--', linewidth=2)
ax.axvline(0, color='green', linestyle=':', linewidth=2, label='Perfect liquidation')

ax.set_xlabel('Final Inventory at $\\tau^{\\mathrm{cl}}$', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Final Inventory Distribution', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("final_evaluation_inventory.png", dpi=150)
plt.show()

print("\n" + "="*70)
print("Evaluation complete! Plots saved.")
print("="*70)