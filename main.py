import math
import itertools
import random
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import torch.optim as optim
import matplotlib.pyplot as plt

from dataclasses import dataclass, field

plt.rcParams['text.usetex'] = True

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
    axs[2,2].plot(tr.t, tr.act_v, label='$v_t$ (CLOB)')
    axs[2,2].plot(tr.t, tr.act_delta, label='$\delta_t$ (CLOB)')
    axs[2,2].plot(tr.t, tr.act_K, label='$K_t^a$ (Auction)')
    axs[2,2].plot(tr.t, tr.act_S, label='$S_t^a$ (Auction)')
    axs[2,2].axvline(tau_op, ls='--', lw=0.8, color='k')
    axs[2,2].legend(); axs[2,2].set_title('Actions')

    for ax in axs.flat:
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Episode {ep} timeline", y=0.995)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

class MarketEmulator:
    def __init__(self, tau_op=10, tau_cl=15, I=10000, V=5000, L=10, Lc=5, La=5,
                 lambda_param=1.0, kappa=0.1, q=1.0, d=1.0, gamma=0.5, 
                 v_m=1000, pareto_gamma=2.0, poisson_rate=0.5, sigma_mid=1.0, seed=None,
                 V_top_max=5000.0, lambda_decision=1.0, alpha=1.0):
        """
        Initialize the market emulator with given parameters.
        """
        if seed is not None:
            torch.manual_seed(seed)
        # Save parameters
        self.tau_op = tau_op
        self.tau_cl = tau_cl
        self.I_max = I
        self.V_max = V
        self.L_max = L
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
        self.V_top_max = V_top_max
        self.lambda_decision = lambda_decision
        self.alpha = float(alpha)

        # Running moments across decision times i for each price tick k
        self._mom_sum = defaultdict(float)     # ∑_{s≤i} vol_s^k
        self._mom_sum_sq = defaultdict(float)  # ∑_{s≤i} (vol_s^k)^2
        self._mom_count = 0    
        # Initialize state
        self.reset()
        
    def _refresh_order_book(self):
    # Fresh, independent Beta draws for ask/bid L1
        V1a = 1000.0 * (4.0 * np.random.beta(0.5, 0.5) + 1.0)
        V1b = 1000.0 * (4.0 * np.random.beta(0.5, 0.5) + 1.0)
        V1a = float(min(V1a, self.V_top_max))
        V1b = float(min(V1b, self.V_top_max))
        # Geometric decay below the top level
        self.ask_volumes = [V1a * (0.5 ** j) for j in range(self.Lc)]
        self.bid_volumes = [V1b * (0.5 ** j) for j in range(self.Lc)]
        # With positive volumes at all levels, depth is full book
        self.depth_ask = self.Lc
        self.depth_bid = self.Lc
    
    def reset(self):
        """Reset the environment to the beginning of an episode."""
        # Phase and time
        self.phase = 'continuous'
        self.current_time = 0.0
        # Inventory and executed volume
        self.inventory = float(self.I_max)
        self.last_executed = 0.0
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
                    process_buy_order(volume=min(5000.0, self.V_max))
                    next_buy_time = current_time + np.random.exponential(scale=1.0 / self.poisson_rate)
                else:
                    current_time = next_sell_time
                    process_sell_order(volume=min(5000.0, self.V_max))
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
                K_new = np.random.uniform(0.0, 50.0)
                S_new = self.mid_price + np.random.uniform(-5.0, 5.0) * 2.0  # width 10 around mid
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
import numpy as np

@dataclass
class GLFTParams:
    A: float              # base arrival intensity
    k: float              # intensity decay per currency unit
    gamma: float          # CARA risk aversion
    sigma: float          # mid-price vol (per step, same units as your env)
    alpha_tick: float     # tick size (currency per tick)
    I_max: int            # inventory cap I
    T: int                # CLOB horizon (integer steps) T = tau_op
    dt: float = 1.0       # time step of emulator (1 if each env step is 1)

    @property
    def k_alpha(self):
        return self.k * self.alpha_tick

    @property
    def alpha_GLFT(self):
        # alpha_GLFT = (k*alpha)*gamma*sigma^2/2
        return self.k_alpha * self.gamma * (self.sigma ** 2) / 2.0

    @property
    def eta_GLFT(self):
        # eta_GLFT = A * (1 + gamma/(k*alpha))^{-(1 + (k*alpha)/gamma)}
        ratio = 1.0 + self.gamma / self.k_alpha
        return self.A * (ratio ** (-(1.0 + self.k_alpha / self.gamma)))

import numpy as np
try:
    from scipy.linalg import expm  # optional fast path
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

class GLFTBenchmark:
    
    def __init__(self, params: GLFTParams, clob_actions, q_name_in_state="inv"):
        self.p = params
        self.clob_actions = clob_actions  # e.g. list[(v, delta_ticks)]
        self.q_name = q_name_in_state

        # filled by precompute()
        self.v = None                     # shape (I_max+1, T+1)
        self.delta_star = None            # shape (I_max+1, T+1) in ticks

        # build a fast index of allowed (v>0) actions per delta grid
        self.allowed_clob = [(i, a) for i, a in enumerate(self.clob_actions) if a[0] > 0]
        self.delta_grid = np.array([a[1] for _, a in self.allowed_clob], dtype=float)

    def _build_M_sell_only(self):
        """
        Build the (I+1)×(I+1) lower-bidiagonal generator M for the sell-only ODE:
            \dot v_q = α q^2 v_q - η v_{q-1}, q>=1;    \dot v_0 = 0
        so that v(t) = exp( -M (T - t) ) 1 enforces v(T)=1 exactly.
        """
        I = self.p.I_max
        M = np.zeros((I + 1, I + 1), dtype=float)
        # diagonal: α q^2 for q>=1; 0 for q=0 (since \dot v_0 = 0)
        q = np.arange(I + 1, dtype=float)
        M[np.arange(I + 1), np.arange(I + 1)] = self.p.alpha_GLFT * (q ** 2)
        M[0, 0] = 0.0
        # subdiagonal: -η linking v_{q-1} into \dot v_q
        for qidx in range(1, I + 1):
            M[qidx, qidx - 1] = -self.p.eta_GLFT
        return M

    def _solve_v_timegrid(self, M):
        """
        Compute v[t, q] = [exp( -M (T - t) ) 1]_q for t=0..T, q=0..I.
        Uses eigendecomposition if SciPy is unavailable; SciPy.expm if present.
        """
        I, T = self.p.I_max, self.p.T
        one = np.ones(I + 1, dtype=float)

        if _HAVE_SCIPY:
            v = np.empty((T + 1, I + 1), dtype=float)
            for t in range(T + 1):
                A = expm(-M * (T - t))
                v[t] = A @ one
            return v

        # Fallback: eigendecomposition (works generically; M is lower-bidiagonal)
        # M may be non-symmetric; allow complex and take real part at the end.
        w, V = np.linalg.eig(M)             # M = V diag(w) V^{-1}
        Vinv = np.linalg.inv(V)
        b = Vinv @ one
        v = np.empty((T + 1, I + 1), dtype=complex)
        for t in range(T + 1):
            e = np.exp(-(T - t) * w)        # elementwise
            v[t] = V @ (e * b)
        v = v.real

        # Numerical hygiene: enforce terminal condition and strict positivity
        v[-1, :] = 1.0
        v = np.clip(v, 1e-15, np.inf)
        return v

    def precompute(self):
        """
        Exact precompute of v and delta* from matrix exponential (no time stepping).
        """
        # 1) Build sell-only M and solve v(t) exactly on the integer grid
        M = self._build_M_sell_only()
        v = self._solve_v_timegrid(M)

        # 2) Sanity: shapes & terminal condition
        assert v.shape == (self.p.T + 1, self.p.I_max + 1)
        # v[T,·] should be 1 exactly (up to numerical tolerance)
        if not np.allclose(v[-1], 1.0, rtol=1e-10, atol=1e-12):
            raise RuntimeError("v(T,·) ≠ 1 after matrix exponential; check M construction.")

        # 3) Store and compute optimal ask deltas (in ticks)
        self.v = v
        self.delta_star = self._compute_deltas(v)
        assert self.delta_star.shape == (self.p.T + 1, self.p.I_max + 1)

    def _compute_deltas(self, v):
        """
        delta_a^*(t,q) = (1/(k*alpha)) log(v_q / v_{q-1}) + (1/gamma) log(1 + gamma/(k*alpha))
        Defined for q >= 1; for q=0 we set delta huge (no meaningful ask if no inventory).
        """
        I, T = self.p.I_max, self.p.T
        delt = np.full((T + 1, I + 1), np.inf, dtype=float)  # time-major
        const = (1.0 / self.p.gamma) * np.log(1.0 + self.p.gamma / self.p.k_alpha)
        inv_kalpha = 1.0 / self.p.k_alpha

        eps = 1e-12
        for t in range(T + 1):
            for q in range(1, I + 1):
                ratio = max(v[t, q] / max(v[t, q - 1], eps), eps)
                delt[t, q] = inv_kalpha * np.log(ratio) + const
            delt[t, 0] = np.inf
        return delt  # shape (T+1, I+1)

    def _state_get_inventory(self, s, env=None):
        """
        Robustly extract inventory q from:
        - dict-like state: keys 'inv','inventory','I','q','qty','position'
        - object-like state: same attribute names
        - tuple/list: search each element
        - fallbacks: env.state then env itself
        """
        CANDIDATE_KEYS = (self.q_name, 'inv', 'inventory', 'I', 'q', 'qty', 'position')

        def try_extract(obj):
            if obj is None:
                return None
            # dict-like
            if isinstance(obj, dict):
                for k in CANDIDATE_KEYS:
                    if k in obj:
                        return obj[k]
            # object-like
            for k in CANDIDATE_KEYS:
                if hasattr(obj, k):
                    return getattr(obj, k)
            return None

        # 1) state directly
        q = try_extract(s)

        # 2) if state is (obs, aux) or list, search elements
        if q is None and isinstance(s, (tuple, list)):
            for elem in s:
                q = try_extract(elem)
                if q is not None:
                    break

        # 3) fall back to env.state, then env
        if q is None and env is not None:
            q = try_extract(getattr(env, 'state', None))
        if q is None and env is not None:
            q = try_extract(env)

        if q is None:
            raise AttributeError(f"Cannot find inventory field; looked for {CANDIDATE_KEYS} in state/env.")

        # sanitize
        try:
            q = int(q)
        except Exception:
            # sometimes q comes as 0-dim np.array
            import numpy as _np
            if isinstance(q, _np.ndarray) and q.shape == ():
                q = int(q.item())
            else:
                q = int(float(q))

        q = max(0, min(q, self.p.I_max))
        return q

    def choose_clob_action_index(self, s, t_idx, env=None):
        """
        Map continuous GLFT delta to the nearest discrete (v, delta) in your CLOB_ACTIONS.
        We pick among actions with v>0 and clamp volume to available inventory.
        """
        q = self._state_get_inventory(s, env=env)
        if q <= 0:
            # No inventory: choose any valid no-trade CLOB action if you have one, else the smallest-volume action.
            # Prefer an explicit 'v=0' action if present:
            for i, a in enumerate(self.clob_actions):
                if a[0] == 0:
                    return i
            # else pick the min-volume ask action (v>0) with largest delta to be conservative
            idx, _ = max(self.allowed_clob, key=lambda p: p[1][1])
            return idx

        t_idx = int(np.clip(t_idx, 0, self.p.T))
        delta_star = self.delta_star[t_idx, min(q, self.p.I_max)]
        if not np.isfinite(delta_star):
            # very conservative: fall back to largest delta
            idx, _ = max(self.allowed_clob, key=lambda p: p[1][1])
            return idx

        # nearest delta in the discrete grid
        j = int(np.argmin(np.abs(self.delta_grid - delta_star)))
        idx, (v, d) = self.allowed_clob[j]

        # clamp volume by inventory
        v_clamped = max(1, min(int(v), q))
        # If the chosen tuple volume differs from v_clamped and you encode volume inside the action itself,
        # try to find an action with same delta and v_clamped; else keep idx.
        # Simple fallback: search actions with closest delta and v<=q, prefer largest v (to speed liquidation)
        candidates = [(i, a) for i, a in self.allowed_clob if a[1] == d and a[0] <= q]
        if candidates:
            idx, _ = max(candidates, key=lambda p: p[1][0])  # largest feasible volume
        return idx

    @staticmethod
    def find_auction_noop_index(auction_actions):
        # Look for an explicit "do nothing" triple, else fallback to index 0
        for i, a in enumerate(auction_actions):
            if a == (0, 0, 0) or getattr(a, "is_noop", False):
                return i
        return 0

# ---------------------
# Replay the same episodes with GLFT liquidation (no trading in auction)
# ---------------------
def run_glft_benchmark_episodes(env, glft: GLFTBenchmark, episode_seeds, auction_actions=None):
    """
    Replays exactly len(episode_seeds) episodes with identical randomness,
    choosing GLFT actions in the CLOB and a no-op in the auction.
    Returns: list of episode returns (cumulated reward).
    """
    bm_returns = []
    
    def _normalize_step_out(out):
        # Accepts (s,r,done), (s,r,done,info), or (s,r,terminated,truncated,info)
        if not isinstance(out, tuple):
            raise TypeError(f"env step returned non-tuple: {type(out)}")
        n = len(out)
        if n == 4:
            s, r, done, info = out
            return s, r, bool(done), info
        if n == 3:
            s, r, done = out
            return s, r, bool(done), {}
        if n == 5:
            s, r, terminated, truncated, info = out
            done = bool(terminated) or bool(truncated)
            return s, r, done, info
        raise ValueError(f"Unsupported step return length {n}. Expected 3, 4, or 5.")

    def _step4(env, pref_attr, action):
        """
        Call env.<pref_attr>(action) if it exists, else env.step(action),
        and normalize the output to (s, r, done, info).
        """
        fn = getattr(env, pref_attr, None)
        out = fn(action) if callable(fn) else env.step(action)
        return _normalize_step_out(out)

    # Resolve an auction no-op index if the env provides one; otherwise we’ll pass (0.,0.,0.)
    AUCTION_NOOP_IDX = None
    if auction_actions is not None:
        AUCTION_NOOP_IDX = GLFTBenchmark.find_auction_noop_index(auction_actions)
        
    def _build_auction_noop(env):
        """
        Return a no-op auction action (c_t, K_a, S_a) where c_t is a zero vector
        of appropriate length. Tries a few env attributes; falls back to [].
        """
        n = 0
        try:
            if hasattr(env, "agent_order_active") and env.agent_order_active is not None:
                n = len(env.agent_order_active)
            elif hasattr(env, "agent_orders") and env.agent_orders is not None:
                n = len(env.agent_orders)
            elif hasattr(env, "A_slots") and env.A_slots is not None:
                n = len(env.A_slots)
        except Exception:
            n = 0
        c_t = [0] * int(n)
        return (0.0, 0.0, c_t)


    for ep, seeds in enumerate(episode_seeds):
        # restore PRNG states
        random.setstate(seeds["py"])
        np.random.set_state(seeds["np"])
        torch.random.set_rng_state(seeds["torch"])

        s = env.reset()
        done = False
        cum_r = 0.0
        t_idx = 0

        while not done:
            phase_before = env.phase

            if phase_before in ("continuous", "C", "clob"):
                a_idx = glft.choose_clob_action_index(s, t_idx, env=env)

                if hasattr(env, "step_clob"):
                    s, r, done, info = _step4(env, "step_clob", a_idx)
                else:
                    v, delta = CLOB_ACTIONS[a_idx]
                    s, r, done, info = _step4(env, "step", (float(v), float(delta)))

            elif phase_before in ("auction", "A"):
                # Build a well-typed no-op: c_t is a zero list
                noop = _build_auction_noop(env)

                if hasattr(env, "step_auction"):
                    s, r, done, info = _step4(env, "step_auction", noop)
                else:
                    s, r, done, info = _step4(env, "step", noop)

            else:
                # Unknown phase: advance safely
                s, r, done, info = _step4(env, "step", (0.0, 0.0, 0.0))

            cum_r += float(r)
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
    tau_op=15,           # a bit more continuous time
    tau_cl=18,
    I=1000,              # smaller inventory bound
    V=500,               # smaller per-order cap
    L=10,
    Lc=10,
    La=10,
    lambda_param=5e-3,   # ↓↓↓ shrink terminal inventory penalty
    kappa=0.1,
    q=0.1,              # wrong-side penalty softer
    d=0.1,              # cancellation cost softer
    gamma=0.5,
    v_m=300,             # pareto scale smaller
    pareto_gamma=2.5,    # lighter tails
    poisson_rate=2.0,    # more arrivals → more fills
    sigma_mid=0.2,       # calmer mid-price
    seed=SEED
)

# Actions: smaller steps
V_CHOICES = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
DELTA_CHOICES = [0, 1, 2]
CLOB_ACTIONS = [(v, d) for v, d in itertools.product(V_CHOICES, DELTA_CHOICES)]

K_CHOICES = [0.0, 1.0, 2.5, 5.0, 7.5, 10.0]
S_OFFSETS = [-1, 0, 1]
AUCT_ACTIONS = [(K, off) for K, off in itertools.product(K_CHOICES, S_OFFSETS)]

# Training tweaks
EPISODES = 200
LR = 5e-2          # a bit smaller
GAMMA = 0.995      # slightly longer credit

# ---------------------
# GLFT params (tune/calibrate as you like)
# ---------------------
GLFT_A       = 1.0        # base hit intensity (arbitrary scale)
GLFT_k       = 1.0        # per-currency decay; *effective* slope is k*alpha in ticks
GLFT_gamma   = 1e-4       # CARA risk aversion
GLFT_sigma   = 1.0        # per-step mid-price vol used in benchmark (match your env)
GLFT_T       = int(getattr(env, "tau_op", 100))     # CLOB horizon; falls back to 100
GLFT_I_MAX   = int(getattr(env, "I", 50))           # inventory cap I
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
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        return self.net(x)

clob_net = DQN(in_dim=8, out_dim=len(CLOB_ACTIONS)).to(DEVICE)
auct_net = DQN(in_dim=7, out_dim=len(AUCT_ACTIONS)).to(DEVICE)

clob_target = DQN(in_dim=8, out_dim=len(CLOB_ACTIONS)).to(DEVICE)
auct_target = DQN(in_dim=7, out_dim=len(AUCT_ACTIONS)).to(DEVICE)
clob_target.load_state_dict(clob_net.state_dict()); clob_target.eval()
auct_target.load_state_dict(auct_net.state_dict()); auct_target.eval()

def soft_update(target, source, tau=0.01):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1 - tau).add_(sp.data, alpha=tau)


opt_clob = optim.Adam(clob_net.parameters(), lr=LR)
opt_auct = optim.Adam(auct_net.parameters(), lr=LR)
REWARD_SCALE = 1e-3  # scale down rewards by 1000
mse = nn.SmoothL1Loss()  # Huber

# ε-greedy schedule
def epsilon_by_episode(ep, start=1.0, end=0.05, total=EPISODES):
    if total <= 1: return end
    frac = ep / (total - 1)
    return start + (end - start) * frac

# ---------------------
# Storage for plots
# ---------------------
all_mid_paths = []
all_step_rewards = []
all_cum_rewards = []
final_inventories = []

eval_returns = []

def run_eval_episode(env, clob_net, auct_net):
    s = env.reset()
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
            K, off = AUCT_ACTIONS[a_idx]
            S = s['X10'] + off
            c_vec = [0] * (env.tau_cl + 1)
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
PLOT_EVERY_N = 50   # 0/None disables the cadence filter

# Collectors for regret analysis
EPISODE_SEEDS = []
DQN_RETURNS = []

for ep in range(EPISODES):
    eps = epsilon_by_episode(ep)
    s = env.reset()
    
    # Save seeds so we can replay the same randomness with the benchmark policy
    EPISODE_SEEDS.append({
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.random.get_rng_state()
    })

    
    tracker = EpisodeTracker()
    mid_track = []
    r_track = []
    cum_track = []
    cum_r = 0.0
    done = False
    
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
            K, off = AUCT_ACTIONS[a_idx]
            S = s['X10'] + off
            c_vec = [0] * (env.tau_cl + 1)

            s2, r, done = env.step((K, S, c_vec))

            if env.phase == 'auction' and not done:
                x2 = feat_auction(s2, env).unsqueeze(0)
                with torch.no_grad():
                    q_next = auct_target(x2).max(dim=1)[0]
            else:
                q_next = torch.tensor([0.0], device=DEVICE)


            target = torch.tensor([r], device=DEVICE) + GAMMA * q_next
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
        DQN_RETURNS.append(cum_r)

        s = s2

    all_mid_paths.append(mid_track)
    all_step_rewards.append(r_track)
    all_cum_rewards.append(cum_track)
    final_inventories.append(env.inventory)
    
    eval_r = run_eval_episode(env, clob_net, auct_net)
    eval_returns.append(eval_r)

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
plt.figure(figsize=(7,4))
for ep, path in enumerate(all_mid_paths, 1):
    plt.plot(path, label=f"Ep {ep}")
plt.title("Mid-price trajectories (decision steps)")
plt.xlabel("Decision step"); plt.ylabel("Mid price")
plt.legend(ncol=2, fontsize=8); plt.tight_layout(); plt.show()

plt.figure(figsize=(7,4))
for ep, rpath in enumerate(all_step_rewards, 1):
    plt.plot(rpath, label=f"Ep {ep}")
plt.title("Step rewards per episode")
plt.xlabel("Decision step"); plt.ylabel("Reward")
plt.legend(ncol=2, fontsize=8); plt.tight_layout(); plt.show()

plt.figure(figsize=(7,4))
for ep, cpath in enumerate(all_cum_rewards, 1):
    plt.plot(cpath, label=f"Ep {ep}")
plt.title("Cumulative reward per episode")
plt.xlabel("Decision step"); plt.ylabel("Cumulative reward")
plt.legend(ncol=2, fontsize=8); plt.tight_layout(); plt.show()

plt.figure(figsize=(6,4))
plt.bar(range(1, EPISODES+1), final_inventories)
plt.axhline(0, color='k', lw=0.7)
plt.title("Final inventory by episode")
plt.xlabel("Episode"); plt.ylabel("Inventory at $\\tau^\mathrm{cl}$")
plt.tight_layout(); plt.show()

print("Final inventories:", final_inventories)
print("Total cumulative rewards:", [round(c[-1], 2) for c in all_cum_rewards])

# ---------------------
# Loss vs episodes
# ---------------------
plt.figure(figsize=(7,4))
plt.plot(range(1, EPISODES+1), clob_loss_per_ep, marker='o', label='CLOB DQN loss (mean/ep)')
plt.plot(range(1, EPISODES+1), auct_loss_per_ep, marker='o', label='Auction DQN loss (mean/ep)')
plt.xlabel("Episode")
plt.ylabel("MSE loss")
plt.title("DQN training loss per episode")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("Mean CLOB losses per episode:", [round(v, 6) for v in clob_loss_per_ep])
print("Mean Auction losses per episode:", [round(v, 6) for v in auct_loss_per_ep])

plt.figure(figsize=(7,4))
plt.plot(range(1, EPISODES+1), eval_returns, marker='o')
plt.title("Evaluation return ($\\varepsilon = 0$) vs episode")
plt.xlabel("Episode"); plt.ylabel("Total return")
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

# ---------------------
# Evaluate GLFT on the same episodes, then compute & plot pseudo-regret
# ---------------------
# If you have a global AUCTION_ACTIONS, pass it so we can find a no-op
# --- right before calling the benchmark ---
n_dqn = len(DQN_RETURNS)
n_seed = len(EPISODE_SEEDS)
N = min(n_dqn, n_seed)

if N == 0:
    raise RuntimeError("No completed episodes to evaluate.")

# replay only the episodes for which we have BOTH seed and DQN return
bm_returns = run_glft_benchmark_episodes(env, glft, EPISODE_SEEDS[:N], auction_actions=None)

# align arrays
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
plt.show()