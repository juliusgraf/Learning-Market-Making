import math
import random
import numpy as np
import torch
from scipy.linalg import expm
import torch.nn as nn
from collections import defaultdict
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Optional
import copy
import pandas as pd
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from dataclasses import dataclass, field
from contextlib import contextmanager

def load_mid_matrix_from_csv(
    csv_path: str,
    tickers=None,
    start=None,
    tau_cl: int = 150,
    normalize_start_100: bool = False,
):
    df = pd.read_csv(csv_path, parse_dates=["Datetime"]).sort_values("Datetime")

    if tickers is None:
        tickers = [c for c in df.columns if c != "Datetime"]

    if start is None:
        start_idx = 0
    elif isinstance(start, int):
        start_idx = start
    else:
        start_ts = pd.Timestamp(start)
        start_idx = int(np.searchsorted(df["Datetime"].to_numpy(), start_ts.to_datetime64()))

    need = tau_cl + 1
    win = df.iloc[start_idx : start_idx + need]
    if len(win) < need:
        raise ValueError(f"Not enough rows: need {need}, have {len(win)} (start_idx={start_idx}).")

    X = win[tickers].to_numpy(dtype=float).T
    if normalize_start_100:
        X = X / X[:, [0]] * 100.0

    times = win["Datetime"].to_numpy()
    return times, tickers, X


@dataclass
class EpisodeTracker:
    t: list = field(default_factory=list)
    phase: list = field(default_factory=list)
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
    act_v: list = field(default_factory=list)
    act_delta: list = field(default_factory=list)
    act_K: list = field(default_factory=list)
    act_S: list = field(default_factory=list)

class MarketEmulator:
    def __init__(self, tau_op=60, tau_cl=70, T = 70, I=100, V=5000, L=10, Lc=8, La=5,
             lambda_param=0.005, kappa=0.1, q=1.0, d=1.0, gamma=0.5, 
             v_m=1000, pareto_gamma=2.0, poisson_rate=0.5, seed=None,
             V_top_max=50000.0, lambda_decision=1.0, alpha=0.01, beta_a=2.0, beta_b=5.0, 
             depth_decay=0.6, price_band_ticks=5.0, L_max_auction=100,
             sigma_mid=0.1, mid_price_path=None):

        self._seed: Optional[int] = None
        self.rng: np.random.Generator = np.random.default_rng()
        if seed is not None:
            self.set_seed(seed)
            
        self.mid_price_path = None
        if mid_price_path is not None:
            self.mid_price_path = np.asarray(mid_price_path, dtype=float)

        if self.mid_price_path is None or len(self.mid_price_path) < 2:
            raise ValueError("mid_price_path must be provided with length >= 2")
        self.mid_price = float(self.mid_price_path[0])

        self.tau_op = tau_op
        self.tau_cl = tau_cl
        self.T = float(T)
        self.dt = self.T / float(self.tau_cl)
        self.I_max = I
        self.V_max = V
        self.L_max_clob = L
        self.L_max = L_max_auction
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

        self._mom_sum = defaultdict(float)
        self._mom_sum_sq = defaultdict(float)
        self._mom_count = 0    

        self.reset()
        
    def set_seed(self, seed: int) -> None:
        s = int(seed)
        self._seed = s
        self.rng = np.random.default_rng(s)
        
    def _update_mid_price_from_path(self) -> None:
        if self.mid_price_path is None or len(self.mid_price_path) == 0:
            return
        idx = int(math.floor(self.current_time))
        idx = max(0, min(idx, len(self.mid_price_path) - 1))
        self.mid_price = float(self.mid_price_path[idx])

        
    def _refresh_order_book(self):
        V1a = float(self.V_top_max * self.rng.beta(self.beta_a, self.beta_b))
        V1b = float(self.V_top_max * self.rng.beta(self.beta_a, self.beta_b))
        rho = self.depth_decay
        self.ask_volumes = [V1a * (rho ** j) for j in range(self.Lc)]
        self.bid_volumes = [V1b * (rho ** j) for j in range(self.Lc)]
        self.depth_ask = self.Lc
        self.depth_bid = self.Lc

    def _sample_mo_volume(self):
        U = self.rng.random()
        vol = self.v_m / ((1.0 - U) ** (1.0 / self.pareto_shape))
        return float(min(vol, self.V_max))
    
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.set_seed(seed)
        self.phase = 'continuous'
        self.current_time = 0.0
        self.inventory = float(self.I_max)
        self.last_executed = 0.0
        self.clob_sell_prices = []
        self.clob_sold_max_price = None

        if self.mid_price_path is not None and len(self.mid_price_path) > 0:
            self.mid_price = float(self.mid_price_path[0])
        else:
            self.mid_price = 100.0
        
        self.mid_price0 = float(self.mid_price)
        
        self.H_cl = self.mid_price
        self._mom_sum.clear()
        self._mom_sum_sq.clear()
        self._mom_count = 0
        if self.mid_price_path is None:
            self._reset_rough_heston_state()
        else:
            self._rh_initialized = False

        self._refresh_order_book() 
        self.N_plus = 0
        self.market_sell_volumes = [0.0] * self.L_max  
        self.market_buy_volumes = [0.0] * self.L_max   
        self.agent_active_order_cont = None

        self.active_supply = []
        self.agent_orders_K = [0.0] * (self.tau_cl + 1)
        self.agent_orders_S = [0.0] * (self.tau_cl + 1)
        self.agent_order_active = [False] * (self.tau_cl + 1)

        self.last_action_time = 0.0
        self.current_time = 0.0

        return self._get_state()
    
    def _get_state(self):
        t = self.current_time
        in_continuous = (self.phase == 'continuous')
        in_auction = (self.phase == 'auction')

        X1 = self.inventory
        X2 = self.last_executed if t == self.tau_cl else 0.0
        X3 = self.H_cl
        X4 = self.depth_ask if in_continuous else 0
        X5 = self.depth_bid if in_continuous else 0
        X6 = len(self.active_supply) if in_auction else 0
        X7 = self.N_plus if in_auction else 0
        X8 = self.N_minus if in_auction else 0
        if in_auction:
            theta = []
            for s in range(self.tau_op, self.tau_cl):
                if self.agent_orders_K[s] != 0 or self.agent_orders_S[s] != 0:
                    canceled = 0 if self.agent_order_active[s] else 1
                else:
                    canceled = 0
                theta.append(canceled)
            theta.append(0)
        else:
            theta = [0] * (self.tau_cl + 1)
        X9 = theta
        X10 = self.mid_price
        if in_auction:
            X11 = [self.market_sell_volumes[i] if i < self.N_plus else 0.0 for i in range(self.L_max)]
            X12 = [self.market_buy_volumes[i]  if i < self.N_minus else 0.0 for i in range(self.L_max)]
        else:
            X11 = [0.0] * self.L_max
            X12 = [0.0] * self.L_max
        if in_continuous:
            X13 = []
            for j in range(self.Lc):
                vol = self.ask_volumes[j] if j < len(self.ask_volumes) else 0.0
                if j+1 > self.depth_ask: 
                    vol = 0.0
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
        B = 10
        X15 = []
        if in_auction:
            for idx in range(self.La):
                if idx < len(self.active_supply):
                    K_i, S_i = self.active_supply[idx]
                    X15.append(K_i)
                    X15.append(S_i)
                else:
                    X15.append([0.0] * 2)
        else:
            X15 = [[0.0] * 2 for _ in range(self.La)]
        X16 = []
        X17 = []
        for s in range(int(self.tau_cl) + 1):
            if self.tau_op <= s < t:
                X16.append(self.agent_orders_S[s])
                X17.append(self.agent_orders_K[s])
            else:
                X16.append(0.0)
                X17.append(0.0)
        return {
            'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5,
            'X6': X6, 'X7': X7, 'X8': X8, 'X9': X9, 'X10': X10,
            'X11': X11, 'X12': X12, 'X13': X13, 'X14': X14,
            'X15': X15, 'X16': X16, 'X17': X17,
            'time': self.current_time
        }
        
    def _update_hyp_clearing_price_from_book(self):
        eps = 1e-12
        tick_size = self.alpha

        k0 = int(round(self.mid_price / tick_size))
        vol_by_k = {}

        for j in range(min(self.Lc, len(self.ask_volumes))):
            v = float(self.ask_volumes[j])
            if v > eps:
                k = k0 + j
                vol_by_k[k] = vol_by_k.get(k, 0.0) + v

        for j in range(min(self.Lc, len(self.bid_volumes))):
            v = float(self.bid_volumes[j])
            if v > eps:
                k = k0 - j
                vol_by_k[k] = vol_by_k.get(k, 0.0) + v
                
        if self.agent_active_order_cont and self.agent_active_order_cont['volume'] > 1e-12:
            j = int(self.agent_active_order_cont['level'])
            k = k0 + j
            vol_by_k[k] = vol_by_k.get(k, 0.0) + float(self.agent_active_order_cont['volume'])

        self._mom_count += 1
        for k, v in vol_by_k.items():
            self._mom_sum[k]    += v
            self._mom_sum_sq[k] += v * v

        num = 0.0
        den = 0.0
        i = self._mom_count
        for k, s in self._mom_sum.items():
            e_hat = s / i
            if e_hat <= eps:
                continue
            sig_hat = self._mom_sum_sq[k] / i
            K_hat = max(0.0, (2.0 * e_hat - sig_hat / max(e_hat, eps)) / tick_size)
            if K_hat > 0.0:
                num += K_hat * (tick_size * k)
                den += K_hat

        if den > 0.0:
            tilde_S = num / den
            self.H_cl = self.H_cl + self.gamma * (tilde_S - self.H_cl)
   
    def step(self, action):

        f = lambda x : x if x > 0 else 0

        if self.phase == 'continuous':
            
            v, delta = action
            v = float(v)
            delta = float(delta)
            if v > self.inventory: 
                v = self.inventory
            v = max(0.0, min(v, self.V_max))
            self.agent_active_order_cont = None
            lam_step = self.poisson_rate * self.dt

            j_agent = int(max(0, min(self.Lc - 1, math.floor(delta))))
            k_mid = math.floor(self.mid_price / self.alpha)
            agent_price = self.alpha * (k_mid + j_agent)

            if v > 0 and j_agent < self.Lc:
                self.agent_active_order_cont = {'level': j_agent, 'price': agent_price, 'volume': v}
            
            last_time = self.current_time
            next_allowed_time = min(math.floor(last_time) + 1, self.tau_op - 1)
            target_time = float(next_allowed_time)

            executed_vol = 0.0

            next_buy_time = last_time + self.rng.exponential(scale=1.0 / lam_step)
            next_sell_time = last_time + self.rng.exponential(scale=1.0 / lam_step)

            def process_buy_order(volume):
                nonlocal executed_vol
                remain = float(volume)

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
            tau_plus  = next_buy_time
            tau_minus = next_sell_time
            tau_i = max(tau_plus, tau_minus)

            t_i = math.floor(last_time) + 1.0

            target_time = min(max(t_i, tau_i), self.tau_op - 1.0)

            while min(next_buy_time, next_sell_time) <= target_time:
                if next_buy_time <= next_sell_time:
                    current_time = next_buy_time
                    process_buy_order(volume=self._sample_mo_volume())
                    next_buy_time = current_time + self.rng.exponential(scale=1.0 / lam_step)
                else:
                    current_time = next_sell_time
                    process_sell_order(volume=self._sample_mo_volume())
                    next_sell_time = current_time + self.rng.exponential(scale=1.0 / lam_step)
            self.depth_ask = next((j+1 for j,v in enumerate(self.ask_volumes) if v <= 1e-6), self.Lc)
            self.depth_bid = next((j+1 for j,v in enumerate(self.bid_volumes) if v <= 1e-6), self.Lc)
            
            self._update_hyp_clearing_price_from_book()

            S_submit = agent_price
            E_t = executed_vol
            reward = S_submit * E_t * f(1 - self.kappa * f(self.H_cl - S_submit))
            
            if E_t > 0.0:
                self.clob_sell_prices.append(float(S_submit))
                if (self.clob_sold_max_price is None) or (float(S_submit) > self.clob_sold_max_price):
                    self.clob_sold_max_price = float(S_submit)

            self.last_executed = E_t
            self.inventory = float(np.clip(self.inventory, -self.I_max, self.I_max))

            dt = max(0.0, target_time - last_time)
            dt_phys  = dt * self.dt

            self.current_time = float(target_time)
            self.last_action_time = self.current_time

            self._refresh_order_book()
            self.agent_active_order_cont = None

            self._update_mid_price_from_path()

            self.next_decision_time = self.current_time + dt

            done = False
            if self.current_time >= self.tau_op - 1:
                self.phase = 'auction'
                self.current_time = float(self.tau_op)
                self.N_plus = 0
                self.N_minus = 0
                self.market_sell_volumes = [0.0] * self.L_max
                self.market_buy_volumes = [0.0] * self.L_max
                self.active_supply = [] 

            next_state = self._get_state()
            return next_state, reward, done
        
        elif self.phase == 'auction':
            K_a, S_a, c_t = action
            K_a = float(K_a); S_a = float(S_a)
            t = int(self.current_time)
            done = False

            # Create new limit order via Bernoulli(0.3)
            if self.rng.random() < 0.3 and len(self.active_supply) < self.La:
                K_new = self.rng.uniform(0.1, 2.0)
                S_new = self.mid_price + self.rng.integers(-int(self.price_band_ticks), int(self.price_band_ticks)+1) * self.alpha
                self.active_supply.append((float(K_new), float(S_new)))
            # Cancel a random existing order via independent Bernoulli(0.2)
            if self.rng.random() < 0.2 and self.active_supply:
                idx = int(self.rng.integers(0, len(self.active_supply)))
                self.active_supply.pop(idx)
            # Create new sell market order via Bernoulli(0.3)
            if self.rng.random() < 0.3:
                U = self.rng.random()
                vol = self.v_m / ((1.0 - U) ** (1.0 / self.pareto_shape))
                vol = float(min(vol, self.V_max))
                if self.N_plus < self.L_max:
                    self.market_sell_volumes[self.N_plus] = vol
                self.N_plus = min(self.N_plus + 1, self.L_max)
            # Create new buymarket order via Bernoulli(0.3)
            if self.rng.random() < 0.3:
                U = self.rng.random()
                vol = self.v_m / ((1.0 - U) ** (1.0 / self.pareto_shape))
                vol = float(min(vol, self.V_max))
                if self.N_minus < self.L_max:
                    self.market_buy_volumes[self.N_minus] = vol
                self.N_minus = min(self.N_minus + 1, self.L_max)
            # Cancel random existing sell market order via independent Bernoulli(0.1)
            if self.rng.random() < 0.1:
                if self.N_plus > 0 and self.rng.random() < 0.5:
                    idx = int(self.rng.integers(0, self.N_plus))
                    self.market_sell_volumes[idx] = 0.0
            # Cancel random existing buy market order via independent Bernoulli(0.1)
            if self.rng.random() < 0.1:
                if self.N_minus > 0 and self.rng.random() < 0.5:
                    idx = int(self.rng.integers(0, self.N_minus))
                    self.market_buy_volumes[idx] = 0.0

            # Apply cancellations from agent action
            if isinstance(c_t, torch.Tensor):
                c_t = c_t.tolist()
            for s in range(self.tau_op, t):
                if s < len(c_t) and c_t[s] == 1 and self.agent_order_active[s]:
                    self.agent_order_active[s] = False

            if K_a > 0:
                self.agent_orders_K[t] = K_a
                self.agent_orders_S[t] = S_a
                self.agent_order_active[t] = True

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

            if isinstance(c_t, (list, tuple)):
                cancel_count = sum(1 for s in range(self.tau_op, t) if s < len(c_t) and c_t[s] == 1)
            reward = K_a * self.H_cl * (self.H_cl - S_a) - self.q * f(- K_a * self.H_cl *(self.H_cl - S_a)) - self.d * cancel_count

            self.last_executed = 0.0
            self.current_time = float(t + 1)

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

                executed_final = 0.0
                for s in range(self.tau_op, self.tau_cl):
                    if self.agent_order_active[s]:
                        executed_final += self.agent_orders_K[s] * (clearing_price - self.agent_orders_S[s])

                I_final = self.inventory - executed_final
                I_final = float(np.clip(I_final, -self.I_max, self.I_max))
                self.inventory = I_final
                self.last_executed = executed_final

                terminal_reward = executed_final * clearing_price- self.lambda_param * (abs(I_final) ** 2)
                wrong_side_sum = 0.0
                for s in range(self.tau_op, self.tau_cl):
                    if self.agent_order_active[s]:
                        wrong_side_sum += f(- self.agent_orders_K[s] * clearing_price * (clearing_price - self.agent_orders_S[s]))
                terminal_reward -= self.q * wrong_side_sum
                reward += terminal_reward
                done = True

            next_state = self._get_state()
            return next_state, reward, done
        
        else:
            return self._get_state(), 0.0, True    
        

@contextmanager
# No wrong-side and cancellation penalties for theoretical benchmark
def benchmark_penalty_override(env, *, disable_wrong_side=True, disable_cancel=True):
    old_q, old_d = float(env.q), float(env.d)
    if disable_wrong_side: env.q = 0.0
    if disable_cancel: env.d = 0.0
    try:
        yield
    finally:
        env.q, env.d = old_q, old_d

@dataclass
class ASParams:
    A: float
    k: float
    gamma: float
    sigma: float
    alpha_tick: float
    I_max: int
    T: int
    dt: float = 1.0

    @property
    def alpha_AS(self) -> float:
        if self.gamma == 0.0:
            return 0.0
        return self.k * self.gamma * (self.sigma ** 2) / 2.0

    @property
    def eta_AS(self) -> float:
        if self.gamma == 0.0:
            return self.A / math.e
        r = 1.0 + self.gamma / self.k
        return self.A * (r ** (-(1.0 + self.k / self.gamma)))

    @property
    def k_per_tick(self) -> float:
        return self.k * self.alpha_tick

    @property
    def inv_k_per_tick(self) -> float:
        return 1.0 / self.k_per_tick

    @property
    def const_ticks(self) -> float:
        if self.gamma == 0.0:
            return 1.0 / (self.k * self.alpha_tick)
        return (1.0 / (self.gamma * self.alpha_tick)) * math.log(1.0 + self.gamma / self.k)

class ASBenchmark:
    def __init__(self, params: ASParams, clob_actions, q_name_in_state="inv"):
        self.p = params
        self.clob_actions = list(clob_actions)
        self.q_name = q_name_in_state

        self.v = None
        self.delta_star = None

        self.allowed_clob = [(i, a) for i, a in enumerate(self.clob_actions) if a[0] > 0]
        self.delta_grid = np.array([a[1] for _, a in self.allowed_clob], dtype=float)

    def _build_M(self):
        I = self.p.I_max
        M = np.zeros((I + 1, I + 1), dtype=float)

        q = np.arange(I + 1, dtype=float)
        M[np.arange(I + 1), np.arange(I + 1)] = self.p.alpha_AS * (q ** 2)
        M[0, 0] = 0.0

        eta = self.p.eta_AS
        for j in range(1, I + 1):
            M[j, j - 1] = -eta

        return M

    def _solve_v(self, M):
        I, T, dt = self.p.I_max, self.p.T, self.p.dt
        E = expm(-M * dt)
        v = np.empty((T + 1, I + 1))
        v[T] = 1.0
        for t in range(T - 1, -1, -1):
            v[t] = E @ v[t + 1]
        return v

    def precompute(self):
        M = self._build_M()
        v = self._solve_v(M)

        assert v.shape == (self.p.T + 1, self.p.I_max + 1)
        if not np.allclose(v[-1], 1.0, rtol=1e-10, atol=1e-12):
            raise RuntimeError("v(T,·) is not 1: terminal condition unsatisfied")

        self.v = v
        self.delta_star = self._compute_deltas(v)
        assert self.delta_star.shape == (self.p.T + 1, self.p.I_max + 1)

    def _compute_deltas(self, v):
        I, T = self.p.I_max, self.p.T
        delt = np.full((T + 1, I + 1), np.inf, dtype=float)
        inv_k_tick = self.p.inv_k_per_tick
        const_ticks = self.p.const_ticks

        eps = 1e-16
        for t in range(T + 1):
            for q in range(1, I + 1):
                ratio = v[t, q] / max(v[t, q - 1], eps)
                ratio = max(ratio, eps)
                delt[t, q] = inv_k_tick * math.log(ratio) + const_ticks
            delt[t, 0] = math.inf
        return delt

    def _state_get_inventory(self, s, env=None):
        CANDIDATE_KEYS = (self.q_name, 'inv', 'inventory', 'I', 'q', 'quantity', 'position')

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
        q = self._state_get_inventory(s, env=env)
        if q <= 0:
            for i, a in enumerate(self.clob_actions):
                if a[0] == 0:
                    return i
            if self.allowed_clob:
                i, _ = max(self.allowed_clob, key=lambda p: p[1][1])
                return i
            return 0

        t_idx = int(np.clip(int(t_idx), 0, self.p.T))
        delta_star = self.delta_star[t_idx, min(q, self.p.I_max)]
        if not np.isfinite(delta_star) or len(self.allowed_clob) == 0:
            idx, _ = max(self.allowed_clob, key=lambda p: p[1][1])
            return idx

        j = int(np.argmin(np.abs(self.delta_grid - delta_star)))
        idx, (v, d) = self.allowed_clob[j]

        candidates = [(i, a) for i, a in self.allowed_clob if a[1] == d and a[0] <= q]
        if candidates:
            idx, _ = max(candidates, key=lambda p: p[1][0])
        return idx
    


class TWAPBenchmark:


    def __init__(
        self,
        params: ASParams,
        clob_actions,
        q_name_in_state: str = "inv",
        twap_delta_ticks: 'Optional[float]' = None,
        delta_mode: str = "min",
    ):
        self.p = params
        self.clob_actions = list(clob_actions)
        self.q_name = q_name_in_state

        self.allowed_clob = [(i, a) for i, a in enumerate(self.clob_actions) if a[0] > 0]
        if len(self.allowed_clob) == 0:
            raise ValueError("TWAPBenchmark: no CLOB actions with positive volume.")

        self.delta_grid = np.array([a[1] for _, a in self.allowed_clob], dtype=float)

        if twap_delta_ticks is not None:
            d = float(twap_delta_ticks)
            j = int(np.argmin(np.abs(self.delta_grid - d)))
            _, (_, d_snap) = self.allowed_clob[j]
            self.twap_delta = float(d_snap)
        else:
            if delta_mode == "max":
                self.twap_delta = float(np.max(self.delta_grid))
            else:
                self.twap_delta = float(np.min(self.delta_grid))

    def _state_get_inventory(self, s, env=None):
        CANDIDATE_KEYS = (self.q_name, 'inv', 'inventory', 'I', 'q', 'quantity', 'position')

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
        q = self._state_get_inventory(s, env=env)

        if q <= 0:
            for i, a in enumerate(self.clob_actions):
                if a[0] == 0:
                    return i
            i, _ = max(self.allowed_clob, key=lambda p: p[1][1])
            return i

        t_idx = int(np.clip(int(t_idx), 0, self.p.T))
        steps_left = max(1, (self.p.T - t_idx + 1))

        target_vol = int(math.ceil(q / float(steps_left)))
        target_vol = max(1, min(target_vol, q))

        tol = 1e-12
        same_delta = [(i, a) for i, a in self.allowed_clob if abs(float(a[1]) - float(self.twap_delta)) <= tol]
        if not same_delta:
            same_delta = list(self.allowed_clob)

        feasible = [(i, a) for i, a in same_delta if float(a[0]) <= float(q)]
        pool = feasible if feasible else same_delta
        if not pool:
            pool = list(self.allowed_clob)

        def score(item):
            i, (v, d) = item
            return (abs(float(v) - float(target_vol)), -float(v))

        idx, _ = min(pool, key=score)
        return int(idx)

def run_twap_benchmark_episodes(env, twap: TWAPBenchmark, episode_seeds, auction_actions=None, ignore_wrong_side=False, ignore_cancel=False):
    bm_returns = []
    with benchmark_penalty_override(env, disable_wrong_side=ignore_wrong_side, disable_cancel=ignore_cancel):

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

        def _extract_time_index(s, env, twap):
            if isinstance(s, dict) and "time" in s:
                t_step = int(round(float(s["time"])))
            else:
                t_step = int(round(float(getattr(env, "current_time", 0.0))))
            return max(0, min(t_step, twap.p.T))

        for seed in episode_seeds:
            s = env.reset(seed=seed)
            done = False
            cum_r = 0.0
            auct_action_open = None

            while not done:
                r_accum = 0.0
                phase_before = getattr(env, "phase", "C")

                if phase_before in ("continuous", "C", "clob"):
                    t_idx = _extract_time_index(s, env, twap)
                    a_idx = twap.choose_clob_action_index(s, t_idx, env=env)

                    if hasattr(env, "step_clob"):
                        s, r, done, info = _step4(env, "step_clob", a_idx)
                    else:
                        v, delta = twap.clob_actions[a_idx]
                        if isinstance(s, dict) and "X1" in s:
                            v = min(float(v), float(s["X1"]))
                        s, r, done, info = _step4(env, "step", (float(v), float(delta)))

                    r_accum += float(r)

                    phase_after = getattr(env, "phase", None)
                    if phase_after in ("auction", "A"):
                        q_left = float(getattr(env, "inventory", 0.0))
                        if q_left < 1e-2:
                            q_left = 0.0
                        if q_left > 0.0 and not done:
                            prices = getattr(env, "clob_sell_prices", None)
                            if prices and len(prices) > 0:
                                pmax = max(prices)
                                pmean = sum(prices) / len(prices)
                                S_star = 0.5 * (pmax + pmean)
                                n = max(1, int(getattr(env, "auction_horizon", 1)))
                                K_big = 10.0 * q_left
                                auct_action = (K_big, S_star, [0.0] * n)
                                s, r, done, info_liq = _step4(env, "step_auction", auct_action)                                     if hasattr(env, "step_auction") else _step4(env, "step", auct_action)
                                r_accum += float(r)

                                if isinstance(info, dict):
                                    info.update({
                                        "auction_opening_order": True,
                                        "I_tau_op": q_left,
                                        "p_max_CLOB": pmax,
                                        "p_mean_CLOB": pmean,
                                        "S_star": S_star
                                    })

                elif phase_before in ("auction", "A"):
                    if auct_action_open is None:
                        n = max(1, int(getattr(env, "auction_horizon", 1)))
                        auct_action_open = (0.0, float(getattr(env, "mid_price", 0.0)), [0.0] * n)
                    K0, S_star, c_t = auct_action_open
                    auct_follow = (0.0, S_star, c_t)
                    s, r, done, info = _step4(env, "step", auct_follow)
                    r_accum += float(r)

                cum_r += r_accum
                t_idx += 1

            bm_returns.append(cum_r)

    return bm_returns


def run_glft_benchmark_episodes(env, glft: ASBenchmark, episode_seeds, auction_actions=None, ignore_wrong_side=False, ignore_cancel=False):
    bm_returns = []
    with benchmark_penalty_override(env, disable_wrong_side=ignore_wrong_side, disable_cancel=ignore_cancel):

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

        def _extract_time_index(s, env, glft):
            if isinstance(s, dict) and "time" in s:
                t_step = int(round(float(s["time"])))
            else:
                t_step = int(round(float(getattr(env, "current_time", 0.0))))
            return max(0, min(t_step, glft.p.T))

        for seed in episode_seeds:
            s = env.reset(seed=seed)
            done = False
            cum_r = 0.0
            auct_action_open = None

            while not done:
                r_accum = 0.0
                phase_before = getattr(env, "phase", "C")

                if phase_before in ("continuous", "C", "clob"):
                    t_idx = _extract_time_index(s, env, glft)
                    a_idx = glft.choose_clob_action_index(s, t_idx, env=env)

                    if hasattr(env, "step_clob"):
                        s, r, done, info = _step4(env, "step_clob", a_idx)
                    else:
                        v, delta = glft.clob_actions[a_idx]
                        s, r, done, info = _step4(env, "step", (float(v), float(delta)))
                    r_accum += float(r)

                    phase_after = getattr(env, "phase", None)
                    if phase_after in ("auction", "A"):
                        q_left = float(getattr(env, "inventory", 0.0))
                        if q_left < 1e-2:
                            q_left = 0.0
                        if q_left > 0.0 and not done:
                            prices = getattr(env, "clob_sell_prices", None)
                            if prices and len(prices) > 0:
                                pmax = max(prices)
                                pmean = sum(prices) / len(prices)
                                S_star = 0.5 * (pmax + pmean)
                                n = max(1, int(getattr(env, "auction_horizon", 1)))
                                K_big = 10.0 * q_left
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
                elif phase_before in ("auction", "A"):
                    if auct_action_open is None:
                        n = max(1, int(getattr(env, "auction_horizon", 1)))
                        auct_action_open = (0.0, float(getattr(env, "mid_price", 0.0)), [0.0] * n)
                    K0, S_star, c_t = auct_action_open
                    auct_follow = (0.0, S_star, c_t)
                    s, r, done, info = _step4(env, "step", auct_follow)
                    r_accum += float(r)

                cum_r += r_accum
                t_idx += 1
            bm_returns.append(cum_r)

    return bm_returns

DEVICE = torch.device("cpu")

def feat_clob(s, env):
    I_max, Lc, Vmax = env.I_max, env.Lc, env.V_max
    t_norm = (s['time'] / max(1.0, (env.tau_op - 1))) if s['time'] <= env.tau_op - 1 else 1.0
    t_norm = float(np.clip(t_norm, 0.0, 1.0))
    top_ask = (s['X13'][0] / Vmax) if len(s['X13']) > 0 and Vmax > 0 else 0.0
    top_bid = (s['X14'][0] / Vmax) if len(s['X14']) > 0 and Vmax > 0 else 0.0
    x_list = [s['X1']/max(1.0, I_max), s['X3'], s['X10'], s['X4']/max(1, Lc), s['X5']/max(1, Lc), top_ask, top_bid, t_norm,]
    return torch.tensor(x_list, dtype=torch.float32, device=DEVICE)

def feat_auction(s, env):
    I_max = env.I_max
    t_norm = (s['time'] - env.tau_op) / max(1.0, (env.tau_cl - env.tau_op))
    t_norm = float(np.clip(t_norm, 0.0, 1.0))
    x_list = [s['X1']/max(1.0, I_max), s['X3'], s['X10'], s['X6']/max(1, env.La), s['X7']/max(1, env.L_max), s['X8']/max(1, env.L_max), t_norm,]
    return torch.tensor(x_list, dtype=torch.float32, device=DEVICE)

# Define the DQN agents for CLOB and auction phases
class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, out_dim)
        )
    def forward(self, x):
        return self.net(x)
    
BUFFER_SIZE = 50000
MIN_BUFFER = 5000
NFQ_EPOCHS = 3
BATCH_SIZE = 128

def nfq_fit_clob(clob_buffer, clob_net, clob_target, auct_target, opt, gamma, mse,
                 batch_size=BATCH_SIZE, epochs=NFQ_EPOCHS):
    N = len(clob_buffer)
    if N == 0:
        return np.nan

    xs = torch.stack([tr[0] for tr in clob_buffer]).to(DEVICE)
    a  = torch.tensor([tr[1] for tr in clob_buffer], dtype=torch.long, device=DEVICE)
    r  = torch.tensor([tr[2] for tr in clob_buffer], dtype=torch.float32, device=DEVICE)
    d  = torch.tensor([1.0 if tr[3] else 0.0 for tr in clob_buffer], dtype=torch.float32, device=DEVICE)
    next_is_auct = torch.tensor([1.0 if tr[4] else 0.0 for tr in clob_buffer], dtype=torch.float32, device=DEVICE)

    
    with torch.no_grad():
        q_next = torch.zeros(N, device=DEVICE)

        idx_cont = (d == 0.0) & (next_is_auct == 0.0)
        idx_auct = (d == 0.0) & (next_is_auct == 1.0)

        if idx_cont.any():
            cont_idx = idx_cont.nonzero(as_tuple=False).squeeze(1).tolist()
            x2_cont = torch.stack([clob_buffer[i][5] for i in cont_idx]).to(DEVICE)
            q_next[idx_cont] = clob_target(x2_cont).max(dim=1)[0]

        if idx_auct.any():
            auct_idx = idx_auct.nonzero(as_tuple=False).squeeze(1).tolist()
            x2_auct = torch.stack([clob_buffer[i][5] for i in auct_idx]).to(DEVICE)
            q_next[idx_auct] = auct_target(x2_auct).max(dim=1)[0]

        y = r + gamma * (1.0 - d) * q_next

    
    total_loss = 0.0
    total_count = 0

    for _ in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        for start in range(0, N, batch_size):
            idx = perm[start:start+batch_size]
            q = clob_net(xs[idx]).gather(1, a[idx].unsqueeze(1)).squeeze(1)
            loss = mse(q, y[idx])

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clob_net.parameters(), 1.0)
            opt.step()

            bs = idx.numel()
            total_loss += float(loss.item()) * bs
            total_count += bs

    return total_loss / max(1, total_count)


def nfq_fit_auction(auct_buffer, auct_net, auct_target, opt, gamma, mse,
                    batch_size=BATCH_SIZE, epochs=NFQ_EPOCHS):
    N = len(auct_buffer)
    if N == 0:
        return np.nan

    xs = torch.stack([tr[0] for tr in auct_buffer]).to(DEVICE)
    a  = torch.tensor([tr[1] for tr in auct_buffer], dtype=torch.long, device=DEVICE)
    r  = torch.tensor([tr[2] for tr in auct_buffer], dtype=torch.float32, device=DEVICE)
    d  = torch.tensor([1.0 if tr[3] else 0.0 for tr in auct_buffer], dtype=torch.float32, device=DEVICE)
    x2 = torch.stack([tr[4] for tr in auct_buffer]).to(DEVICE)

    with torch.no_grad():
        q_next = auct_target(x2).max(dim=1)[0]
        y = r + gamma * (1.0 - d) * q_next

    total_loss = 0.0
    total_count = 0

    for _ in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        for start in range(0, N, batch_size):
            idx = perm[start:start+batch_size]
            q = auct_net(xs[idx]).gather(1, a[idx].unsqueeze(1)).squeeze(1)
            loss = mse(q, y[idx])

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(auct_net.parameters(), 1.0)
            opt.step()

            bs = idx.numel()
            total_loss += float(loss.item()) * bs
            total_count += bs

    return total_loss / max(1, total_count)

mse = nn.SmoothL1Loss()

def epsilon_by_episode(ep, start=1.0, end=0.01, total=None, warmup=100):
    if total <= 1: 
        return end 
    if ep < warmup: return start
    decay_rate = -np.log(end / start) / (total - warmup)
    return start * np.exp(-decay_rate * (ep - warmup))

def run_eval_episode(env, clob_net, auct_net, clob_actions, auct_actions, seed=None):
    s = env.reset(seed=seed)
    done = False
    total_r = 0.0

    while not done:
        if env.phase == 'continuous':
            x = feat_clob(s, env).unsqueeze(0)
            with torch.no_grad():
                a_idx = int(torch.argmax(clob_net(x)[0]).item())
            v, delta = clob_actions[a_idx]
            v = min(v, s['X1'])
            s, r, done = env.step((v, delta))
        else:
            x = feat_auction(s, env).unsqueeze(0)
            with torch.no_grad():
                a_idx = int(torch.argmax(auct_net(x)[0]).item())
            K, off, cancel = auct_actions[a_idx]
            S = s['X10'] + off * env.alpha

            if cancel:
                c_vec = [0] * (env.tau_cl + 1)
                t = int(s['time'])
                if t > env.tau_op:
                    c_vec[env.tau_op:t] = [1] * (t - env.tau_op)
            else:
                c_vec = [0] * (env.tau_cl + 1)

            s, r, done = env.step((K, S, c_vec))

        total_r += float(r)

    return float(total_r)

def estimate_K_from_env(env, n_samples: int = 5000, q_min: float = None) -> float:
    alpha = float(env.alpha)
    Lc = int(env.Lc)
    eps = 1e-9

    lnQ_samples = []
    dP_samples = []

    for _ in range(n_samples):
        env._refresh_order_book()
        Q = float(env._sample_mo_volume())
        if q_min is not None and Q < q_min:
            continue

        ask = [float(v) for v in env.ask_volumes]
        best_before_level = 0
        remain = Q
        j = 0
        while j < Lc and remain > eps:
            lvl = ask[j]
            if lvl > eps:
                take = min(remain, lvl)
                ask[j] -= take
                remain -= take
            j += 1

        best_after_level = None
        for j in range(Lc):
            if ask[j] > eps:
                best_after_level = j
                break
        if best_after_level is None:
            continue

        delta_p = alpha * (best_after_level - best_before_level)
        if delta_p <= 0.0:
            continue

        lnQ_samples.append(math.log(Q))
        dP_samples.append(delta_p)

    if len(lnQ_samples) < 2:
        raise RuntimeError("Not enough samples to estimate K.")

    x = np.asarray(lnQ_samples, dtype=float)
    y = np.asarray(dP_samples, dtype=float)

    K_hat = float(np.dot(x, y) / np.dot(y, y))
    return K_hat

def evaluate_policy(env, clob_net, auct_net, eval_seeds, clob_actions, auct_actions, policy_name="Policy"):
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
                v, delta = clob_actions[a_idx]
                v = min(v, s['X1'])
                s, r, done = env.step((v, delta))
                clob_r += r
            else:
                x = feat_auction(s, env).unsqueeze(0)
                with torch.no_grad():
                    a_idx = int(torch.argmax(auct_net(x)[0]).item())
                K, off, cancel = auct_actions[a_idx]
                S = s['X10'] + off * env.alpha
                if cancel:
                    c_vec = [0]*(env.tau_cl+1)
                    t = int(s['time'])
                    c_vec[env.tau_op:t] = [1]*(t-env.tau_op)
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
    
def soft_update(local_model, target_model, tau=0.01):

    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def train_and_evaluate_one_asset(
    symbol: str,
    mid_path: np.ndarray,
    *,
    start_from_100: bool = True,
    tau_op: int = 120,
    tau_cl: int = 150,
    T: int = 150,
    episodes: int = 1000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    reward_scale: float = 1,
    buffer_size: int = 50000,
    min_buffer: int = 5000,
    batch_size: int = 128,
    nfq_epochs: int = 3,
    eval_interval: int = 100,
    n_eval_episodes: int = 8,
    n_final_eval_episodes: int = 100,
    master_seed: int = 42,
    eval_master_seed: int = 120,
    final_eval_master_seed: int = 200,
    make_plots: bool = False,
    out_dir: str = "results",
) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    mid_path = np.asarray(mid_path, dtype=float)
    if len(mid_path) < 2:
        raise ValueError(f"{symbol}: mid_path must have length >= 2")
    if start_from_100:
        mid_path = 100.0 * mid_path / mid_path[0]

    if len(mid_path) != tau_op:
        raise ValueError(f"{symbol}: len(mid_path)={len(mid_path)} must equal tau_op={tau_op}")

    random.seed(master_seed)
    np.random.seed(master_seed)
    torch.manual_seed(master_seed)
    policy_rng = random.Random(master_seed)

    env = MarketEmulator(
        tau_op=tau_op, tau_cl=tau_cl, T=T,
        I=100, V=30, L=12, Lc=12, La=12,
        L_max_auction=100,
        lambda_param=0.5, kappa=0.1, q=1.0, d=0.1,
        gamma=gamma, v_m=2.0, pareto_gamma=2.5,
        poisson_rate=60*1.0, alpha=0.01,
        seed=None,
        V_top_max=15.0,
        beta_a=2.0, beta_b=5.0, depth_decay=0.5,
        price_band_ticks=10.0,
        mid_price_path=mid_path
    )

    V_CHOICES = np.arange(0, env.V_max + 1, dtype=np.int64)
    DELTA_CHOICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    clob_actions = [(0.0, 0)] + [(float(v), float(d)) for v in V_CHOICES[1:] for d in DELTA_CHOICES]

    STEPS_AUCT = max(1, env.tau_cl - env.tau_op)
    env.auction_horizon = STEPS_AUCT
    K_MAX = 10.0 * env.I_max / STEPS_AUCT
    K_CHOICES = [0.0] + list(np.linspace(1.0, K_MAX, 10))
    S_OFFSETS = np.arange(-12, 13, dtype=int)
    auct_actions = [(float(K), float(off), int(cancel)) for K in K_CHOICES for off in S_OFFSETS for cancel in (0, 1)]

    clob_net = DQN(in_dim=8, out_dim=len(clob_actions)).to(DEVICE)
    auct_net = DQN(in_dim=7, out_dim=len(auct_actions)).to(DEVICE)
    clob_target = DQN(in_dim=8, out_dim=len(clob_actions)).to(DEVICE)
    auct_target = DQN(in_dim=7, out_dim=len(auct_actions)).to(DEVICE)

    clob_target.load_state_dict(clob_net.state_dict()); clob_target.eval()
    auct_target.load_state_dict(auct_net.state_dict()); auct_target.eval()

    initial_clob_state = copy.deepcopy(clob_net.state_dict())
    initial_auct_state = copy.deepcopy(auct_net.state_dict())

    opt_clob = optim.Adam(clob_net.parameters(), lr=lr)
    opt_auct = optim.Adam(auct_net.parameters(), lr=lr)
    mse = nn.SmoothL1Loss()

    from collections import deque
    clob_buffer = deque(maxlen=buffer_size)
    auct_buffer = deque(maxlen=buffer_size)

    def epsilon_by_episode(ep, start=1.0, end=0.01, total=episodes, warmup=100):
        if total <= 1:
            return end
        if ep < warmup:
            return start
        decay_rate = math.log(start / end) / max(1, (total - warmup))
        return start * math.exp(-decay_rate * (ep - warmup))

    rng_train = np.random.default_rng(master_seed)
    EPISODE_SEEDS = rng_train.integers(0, 2**32 - 1, size=episodes, dtype=np.uint32).tolist()

    rng_eval = np.random.default_rng(eval_master_seed)
    EVAL_SEEDS = rng_eval.integers(0, 2**32 - 1, size=n_eval_episodes, dtype=np.uint32).tolist()

    rng_final = np.random.default_rng(final_eval_master_seed)
    FINAL_EVAL_SEEDS = rng_final.integers(0, 2**32 - 1, size=n_final_eval_episodes, dtype=np.uint32).tolist()

    dqn_returns = []
    eval_returns = []
    clob_loss_per_ep = []
    auct_loss_per_ep = []

    for ep in range(episodes):
        seed = int(EPISODE_SEEDS[ep])
        eps = float(epsilon_by_episode(ep))
        s = env.reset(seed=seed)
        done = False
        cum_r = 0.0

        while not done:
            phase_before = env.phase

            if phase_before == 'continuous':
                x = feat_clob(s, env).unsqueeze(0)
                with torch.no_grad():
                    q_vals = clob_net(x)[0]
                a_idx = policy_rng.randrange(len(clob_actions)) if policy_rng.random() < eps else int(torch.argmax(q_vals).item())

                v, delta = clob_actions[a_idx]
                v = min(v, s['X1'])

                s2, r, done = env.step((v, delta))
                next_is_auct = (env.phase != 'continuous')

                
                x2_feat = feat_auction(s2, env) if next_is_auct else feat_clob(s2, env)

                clob_buffer.append((
                    x.squeeze(0).detach(),
                    int(a_idx),
                    float(r * reward_scale),
                    bool(done),
                    bool(next_is_auct),
                    x2_feat.detach()
                ))

            else:
                x = feat_auction(s, env).unsqueeze(0)
                with torch.no_grad():
                    q_vals = auct_net(x)[0]
                a_idx = policy_rng.randrange(len(auct_actions)) if policy_rng.random() < eps else int(torch.argmax(q_vals).item())

                K, off, cancel = auct_actions[a_idx]
                S = float(s['X10'] + off * env.alpha)

                if cancel:
                    c_vec = [0] * (env.tau_cl + 1)
                    t = int(s['time'])
                    if t > env.tau_op:
                        c_vec[env.tau_op:t] = [1] * (t - env.tau_op)
                else:
                    c_vec = [0] * (env.tau_cl + 1)

                s2, r, done = env.step((K, S, c_vec))
                x2 = feat_auction(s2, env)

                auct_buffer.append((
                    x.squeeze(0).detach(),
                    int(a_idx),
                    float(r * reward_scale),
                    bool(done),
                    x2.detach()
                ))

            cum_r += float(r)
            s = s2

        dqn_returns.append(float(cum_r))

        
        if len(clob_buffer) >= min_buffer:
            
            loss_c = nfq_fit_clob(
                clob_buffer, clob_net, clob_target, auct_target,
                opt_clob, gamma, mse, batch_size=batch_size, epochs=nfq_epochs
            )
        else:
            loss_c = np.nan

        if len(auct_buffer) >= min_buffer:
            loss_a = nfq_fit_auction(
                auct_buffer, auct_net, auct_target,
                opt_auct, gamma, mse, batch_size=batch_size, epochs=nfq_epochs
            )
        else:
            loss_a = np.nan
            
        clob_target.load_state_dict(clob_net.state_dict())
        auct_target.load_state_dict(auct_net.state_dict())

        clob_loss_per_ep.append(loss_c)
        auct_loss_per_ep.append(loss_a)

        if (ep + 1) % eval_interval == 0:
            eval_r = float(np.mean([
                run_eval_episode(env, clob_net, auct_net, clob_actions, auct_actions, seed=int(sd))
                for sd in EVAL_SEEDS
            ]))
            eval_returns.append(eval_r)
            print(f"Episode {ep+1}/{episodes} | Eval Return: {eval_r:.2f} | Epsilon: {eps:.3f}")
        else:
            eval_returns.append(eval_returns[-1] if len(eval_returns) else np.nan)

    p = np.asarray(env.mid_price_path, dtype=float)
    r = np.diff(np.log(p))
    AS_sigma = float(np.std(r, ddof=1)) / math.sqrt(float(env.dt))

    AS_A = env.poisson_rate / env.pareto_shape
    AS_k = env.pareto_shape * estimate_K_from_env(env, n_samples=10000)
    AS_gamma = 0.0

    AS_T = int(env.tau_op) - 1
    AS_I_MAX = int(env.I_max)
    AS_ALPHA = float(env.alpha)
    AS_dt = float(env.dt)

    glft_params = ASParams(
        A=float(AS_A),
        k=float(AS_k),
        gamma=float(AS_gamma),
        sigma=float(AS_sigma),
        alpha_tick=float(AS_ALPHA),
        I_max=int(AS_I_MAX),
        T=int(AS_T),
        dt=float(AS_dt),
    )
    glft = ASBenchmark(glft_params, clob_actions, q_name_in_state="inv")
    glft.precompute()

    twap = TWAPBenchmark(glft_params, clob_actions, q_name_in_state="inv", twap_delta_ticks=None, delta_mode="min")

    glft_results = run_glft_benchmark_episodes(
        env, glft, FINAL_EVAL_SEEDS,
        auction_actions=None,
        ignore_wrong_side=False,
        ignore_cancel=False
    )
    glft_mean = float(np.mean(glft_results))

    twap_results = run_twap_benchmark_episodes(
        env, twap, FINAL_EVAL_SEEDS,
        auction_actions=None,
        ignore_wrong_side=False,
        ignore_cancel=False
    )
    twap_mean = float(np.mean(twap_results))

    final_stats = evaluate_policy(env, clob_net, auct_net, FINAL_EVAL_SEEDS, clob_actions, auct_actions, "Final DQN")
    final_mean = float(final_stats["mean_return"])

    initial_clob = DQN(in_dim=8, out_dim=len(clob_actions)).to(DEVICE)
    initial_auct = DQN(in_dim=7, out_dim=len(auct_actions)).to(DEVICE)
    initial_clob.load_state_dict(initial_clob_state); initial_clob.eval()
    initial_auct.load_state_dict(initial_auct_state); initial_auct.eval()

    init_stats = evaluate_policy(env, initial_clob, initial_auct, FINAL_EVAL_SEEDS, clob_actions, auct_actions, "Initial DQN")
    init_mean = float(init_stats["mean_return"])
    improvement_final_minus_glft = final_mean - glft_mean
    improvement_final_minus_twap = final_mean - twap_mean
    twap_minus_glft = twap_mean - glft_mean

    if make_plots:
        
        plt.figure(figsize=(7, 4))
        plt.plot(eval_returns, label="DQN eval mean return")
        plt.title(f"{symbol} - DQN eval return (every {eval_interval} eps)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{symbol}_dqn_eval_returns.png"), dpi=150)
        plt.close()

    return {
        "symbol": symbol,
        "glft_sigma": AS_sigma,
        "glft_A": AS_A,
        "glft_k": AS_k,
        "mean_return_glft": glft_mean,
        "mean_return_twap": twap_mean,
        "mean_return_final_dqn": final_mean,
        "mean_return_initial_dqn": init_mean,
        "improvement_final_minus_glft": improvement_final_minus_glft,
        "improvement_final_minus_twap": improvement_final_minus_twap,
        "twap_minus_glft": twap_minus_glft,
    }

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    CSV_PATH = "data.csv"

    tau_op = 120
    tau_cl = 150
    T = tau_cl

    # SYMBOLS = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "XOM", "JNJ", "PG", "CAT", "NEE"]
    SYMBOLS = ["MSFT", "JPM", "PG", "GOOGL", "CAT"]
    START_ROW = 0

    df = pd.read_csv(CSV_PATH, parse_dates=["Datetime"]).sort_values("Datetime").reset_index(drop=True)

    missing = [s for s in SYMBOLS if s not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")

    if START_ROW + tau_op > len(df):
        raise ValueError(f"Not enough rows: need START_ROW+tau_op <= {len(df)}")

    mid_paths = {
        sym: df[sym].to_numpy(dtype=float)[START_ROW:START_ROW + tau_op]
        for sym in SYMBOLS
    }

    rows = []
    for sym in SYMBOLS:
        res = train_and_evaluate_one_asset(
            sym,
            mid_paths[sym],
            start_from_100=False,
            tau_op=tau_op,
            tau_cl=tau_cl,
            T=T,
            episodes=1000,
            make_plots=False,
        )
        rows.append(res)

    out = pd.DataFrame(rows).sort_values("improvement_final_minus_glft", ascending=False)
    print(out)
    print("Mean improvement vs AS:", out["improvement_final_minus_glft"].mean())
    print("Mean improvement vs TWAP:", out["improvement_final_minus_twap"].mean())
    out.to_csv("sp500_dqn_vs_glft.csv", index=False)
