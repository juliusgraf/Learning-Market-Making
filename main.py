import math
import itertools
import random
import numpy as np
import torch
import torch.nn as nn
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
                 V_top_max=5000.0):
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
        price_grid = [self.mid_price + i for i in range(-B, B+1)]
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
            agent_price = self.mid_price + float(j_agent)  # 1 tick = 1.0

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
            
            current_time = last_time

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
                    
            self._refresh_order_book()

            # Decide at the boundary (always advance at least to target_time)
            next_decision_time = target_time

            # Update order book depths after processing events
            self.depth_ask = next((j+1 for j,v in enumerate(self.ask_volumes) if v <= 1e-6), self.Lc)
            self.depth_bid = next((j+1 for j,v in enumerate(self.bid_volumes) if v <= 1e-6), self.Lc)

            # Update hypothetical clearing price H_t^cl (smoothly toward mid-price)
            self.H_cl += self.gamma * (self.mid_price - self.H_cl)

            # Compute reward for this step
            S_submit = agent_price  # agent's limit price
            E_t = executed_vol       # executed volume of agent's order
            reward = S_submit * E_t * f(1 - self.kappa * f(self.H_cl - S_submit))

            # Update state variables
            self.last_executed = E_t
            self.inventory = float(np.clip(self.inventory, -self.I_max, self.I_max))

            # Advance current time to the next decision time
            dt = max(0.0, next_decision_time - last_time)
            self.current_time = float(next_decision_time)
            self.last_action_time = float(next_decision_time)

            # Remove any remaining agent order (it will be replaced or canceled at next action)
            self.agent_active_order_cont = None

            # Update mid-price via Brownian motion for the time interval
            if dt > 0:
                price_change = float(np.random.normal(loc=0.0, scale=self.sigma_mid * math.sqrt(dt)))
                self.mid_price += price_change

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
# Config
# ---------------------

SEED = 123
DEVICE = torch.device("cpu")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Create env (stable-ish defaults)
env = MarketEmulator(
    tau_op=12,           # a bit more continuous time
    tau_cl=18,
    I=1000,              # smaller inventory bound
    V=500,               # smaller per-order cap
    L=8,
    Lc=6,
    La=5,
    lambda_param=1e-4,   # ↓↓↓ shrink terminal inventory penalty
    kappa=0.1,
    q=0.01,              # wrong-side penalty softer
    d=0.01,              # cancellation cost softer
    gamma=0.5,
    v_m=200,             # pareto scale smaller
    pareto_gamma=2.5,    # lighter tails
    poisson_rate=1.5,    # more arrivals → more fills
    sigma_mid=0.2,       # calmer mid-price
    seed=SEED
)

# Actions: smaller steps
V_CHOICES = [0, 100, 250, 500]
DELTA_CHOICES = [0, 1, 2, 3]
CLOB_ACTIONS = [(v, d) for v, d in itertools.product(V_CHOICES, DELTA_CHOICES)]

K_CHOICES = [0.0, 1.0, 2.5, 5.0, 10.0]
S_OFFSETS = [-1, 0, 1]
AUCT_ACTIONS = [(K, off) for K, off in itertools.product(K_CHOICES, S_OFFSETS)]

# Training tweaks
EPISODES = 5
LR = 5e-3          # a bit smaller
GAMMA = 0.995      # slightly longer credit

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
opt_clob = optim.Adam(clob_net.parameters(), lr=LR)
opt_auct = optim.Adam(auct_net.parameters(), lr=LR)
mse = nn.MSELoss()

# ε-greedy schedule
def epsilon_by_episode(ep, start=0.9, end=0.1, total=EPISODES):
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

# ---------------------
# Training loop
# ---------------------
PLOT_EVERY_EPISODE = True   # ← toggle
SAVE_EP_FIGS = False        # set a path like f"plots/ep_{ep}.png" if you want files

for ep in range(EPISODES):
    eps = epsilon_by_episode(ep)
    s = env.reset()

    tracker = EpisodeTracker()
    mid_track = []
    r_track = []
    cum_track = []
    cum_r = 0.0
    done = False

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
                    q_next = clob_net(x2).max(dim=1)[0]
            else:
                if not done:
                    x2 = feat_auction(s2, env).unsqueeze(0)
                    with torch.no_grad():
                        q_next = auct_net(x2).max(dim=1)[0]
                else:
                    q_next = torch.tensor([0.0], device=DEVICE)

            target = torch.tensor([r], device=DEVICE) + GAMMA * q_next
            q_sa = clob_net(x)[0, a_idx].unsqueeze(0)
            loss = mse(q_sa, target.detach())
            opt_clob.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(clob_net.parameters(), 1.0)
            opt_clob.step()

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
                    q_next = auct_net(x2).max(dim=1)[0]
            else:
                q_next = torch.tensor([0.0], device=DEVICE)

            target = torch.tensor([r], device=DEVICE) + GAMMA * q_next
            q_sa = auct_net(x)[0, a_idx].unsqueeze(0)
            loss = mse(q_sa, target.detach())
            opt_auct.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(auct_net.parameters(), 1.0)
            opt_auct.step()

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

    all_mid_paths.append(mid_track)
    all_step_rewards.append(r_track)
    all_cum_rewards.append(cum_track)
    final_inventories.append(env.inventory)

    if PLOT_EVERY_EPISODE:
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