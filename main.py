import torch, math

class MarketEmulator:
    def __init__(self, tau_op=10, tau_cl=15, I=10000, V=5000, L=10, Lc=5, La=5,
                 lambda_param=1.0, kappa=0.1, q=1.0, d=1.0, gamma=0.5, 
                 v_m=1000, pareto_gamma=2.0, poisson_rate=0.5, sigma_mid=1.0, seed=None):
        """
        Initialize the market emulator with given parameters.
        - tau_op, tau_cl: start and end times of the auction (continuous phase is [0, tau_op-1]).
        - I: initial inventory (and bound).
        - V: volume bound for any single order.
        - L: bound on number of market orders (takers) on each side.
        - Lc: order book depth (number of price levels with initial volume).
        - La: max number of exogenous supply functions in auction.
        - lambda_param, kappa, q, d: reward penalty parameters as defined in the model.
        - gamma: smoothing parameter for clearing price updates in continuous phase.
        - v_m, pareto_gamma: parameters for Pareto distribution of auction market order volumes.
        - poisson_rate: Poisson process rate for market order arrivals in continuous phase.
        - sigma_mid: volatility scale for mid-price Brownian motion.
        - seed: random seed for reproducibility.
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
        # Initialize state
        self.reset()
    
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
        # Sample best level volume from Beta(0.5, 0.5) scaled to [1000, 5000]
        beta_dist = torch.distributions.Beta(torch.tensor([0.5]), torch.tensor([0.5]))
        V1 = 1000 * (4 * beta_dist.sample().item() + 1)
        V1 = float(min(V1, self.V_max))
        # Geometric volume decay for deeper levels
        self.ask_volumes = [V1 * (0.5 ** j) for j in range(self.Lc)]
        self.bid_volumes = [V1 * (0.5 ** j) for j in range(self.Lc)]
        # Determine initial depth L_t^+ and L_t^- (first empty level, if any)
        self.depth_ask = next((j+1 for j,v in enumerate(self.ask_volumes) if v <= 1e-6), self.Lc)
        self.depth_bid = next((j+1 for j,v in enumerate(self.bid_volumes) if v <= 1e-6), self.Lc)
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
            # theta_t^(s) = 1 if order at time s (tau_op <= s < t) was canceled by now
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
        # Return state as a dictionary (could also be a concatenated list/tensor as needed)
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
        
        # Continuous phase step
        if self.phase == 'continuous':
            # Expect action = (v, delta) for volume and price offset (in ticks)
            v, delta = action
            v = float(v); delta = float(delta)
            # Ensure action is admissible
            if v > self.inventory: 
                v = self.inventory
            # If an old order is still active (leftover), cancel it (no cost in continuous)
            if self.agent_active_order_cont:
                self.agent_active_order_cont = None
            # Determine agent's limit price and tick level
            agent_price = self.mid_price + delta * 1.0  # alpha = 1 for tick size
            # Prevent placing sell order below current best bid (approximate by mid as reference)
            if agent_price < self.mid_price:
                agent_price = self.mid_price
            # Determine agent's price level index relative to mid (0 = at mid)
            j_agent = 0
            if agent_price > self.mid_price:
                # number of ticks above mid (assuming 1 tick = 1 price unit)
                j_agent = math.floor((agent_price - self.mid_price) + 0.5)
            # Place agent's order in the OB
            self.agent_active_order_cont = {'level': j_agent, 'price': agent_price, 'volume': v}
            
            # Simulate random market order arrivals until next agent action time
            last_time = self.current_time
            next_allowed_time = math.floor(last_time) + 1
            if next_allowed_time > self.tau_op - 1:
                next_allowed_time = self.tau_op - 1
            target_time = float(next_allowed_time)
            arrived_buy = False  # at least one buy MO arrived
            arrived_sell = False  # at least one sell MO arrived
            executed_vol = 0.0   # total executed volume of agent's order in this interval
            
            # Sample next arrival times for buy and sell Poisson processes
            next_buy_time = last_time + torch.distributions.Exponential(self.poisson_rate).sample().item()
            next_sell_time = last_time + torch.distributions.Exponential(self.poisson_rate).sample().item()
            
            # Define event processing for a buy market order (arriving at next_buy_time)
            def process_buy_order(volume):
                nonlocal executed_vol
                remain = volume
                # 1. Consume ask-side volumes up to agent's price level
                for j in range(min(j_agent, self.Lc)):
                    if remain <= 0:
                        break
                    if self.ask_volumes[j] <= 0:
                        continue
                    if remain >= self.ask_volumes[j]:
                        # consume entire level j
                        remain -= self.ask_volumes[j]
                        self.ask_volumes[j] = 0.0
                    else:
                        # partial fill at level j
                        self.ask_volumes[j] -= remain
                        remain = 0
                        break
                # 2. If buy volume still remains, it will reach agent's level
                if remain > 0 and self.agent_active_order_cont:
                    # Execute agent's order at j_agent with priority
                    vol_agent = self.agent_active_order_cont['volume']
                    if remain >= vol_agent:
                        # entire agent order executed
                        executed_vol += vol_agent
                        self.inventory -= vol_agent
                        remain -= vol_agent
                        # agent order fully filled, remove it
                        self.agent_active_order_cont = None
                    else:
                        # agent order partially executed
                        executed_vol += remain
                        self.inventory -= remain
                        self.agent_active_order_cont['volume'] = vol_agent - remain
                        remain = 0
                # 3. If buy volume remains after agent (or if agent had no order or was filled), consume exogenous vol at agent level and deeper
                if remain > 0:
                    # Consume exogenous volume at agent's price level
                    if j_agent < self.Lc:
                        if remain >= self.ask_volumes[j_agent]:
                            remain -= self.ask_volumes[j_agent]
                            self.ask_volumes[j_agent] = 0.0
                        else:
                            self.ask_volumes[j_agent] -= remain
                            remain = 0
                    # Continue to deeper levels j > j_agent
                    j = j_agent + 1
                    while remain > 0 and j < self.Lc:
                        if self.ask_volumes[j] <= 0:
                            j += 1
                            continue
                        if remain >= self.ask_volumes[j]:
                            remain -= self.ask_volumes[j]
                            self.ask_volumes[j] = 0.0
                        else:
                            self.ask_volumes[j] -= remain
                            remain = 0
                        j += 1
            
            # Define event processing for a sell market order
            def process_sell_order(volume):
                remain = volume
                # Consume bid-side volumes from best bid downward
                for j in range(self.Lc):
                    if remain <= 0:
                        break
                    if self.bid_volumes[j] <= 0:
                        continue
                    if remain >= self.bid_volumes[j]:
                        remain -= self.bid_volumes[j]
                        self.bid_volumes[j] = 0.0
                    else:
                        self.bid_volumes[j] -= remain
                        remain = 0
                        break
            
            current_time = last_time
            # Phase 1: simulate events up to the discrete target_time
            while min(next_buy_time, next_sell_time) <= target_time:
                if next_buy_time <= next_sell_time:
                    # Buy order arrives
                    current_time = next_buy_time
                    arrived_buy = True
                    process_buy_order(volume=min(5000, self.V_max))  # each MO volume = 5000 (capped by V_max)
                    # sample next arrival on buy side
                    next_buy_time = current_time + torch.distributions.Exponential(self.poisson_rate).sample().item()
                else:
                    # Sell order arrives
                    current_time = next_sell_time
                    arrived_sell = True
                    process_sell_order(volume=min(5000, self.V_max))
                    next_sell_time = current_time + torch.distributions.Exponential(self.poisson_rate).sample().item()
            
            # Phase 2: if one side had no arrivals by target_time, wait for the first arrival(s) after target
            next_decision_time = target_time
            if not arrived_buy or not arrived_sell:
                # Extend simulation beyond target_time
                # Determine which sides are missing
                missing_buy = not arrived_buy
                missing_sell = not arrived_sell
                # If both missing, we need two arrivals (one on each side) after target
                if missing_buy and missing_sell:
                    # Wait for first arrival after target (whichever side)
                    first_arrival_time = min(next_buy_time, next_sell_time)
                    if first_arrival_time <= self.tau_op - 1:
                        if next_buy_time <= next_sell_time:
                            current_time = next_buy_time
                            arrived_buy = True
                            process_buy_order(volume=min(5000, self.V_max))
                            next_buy_time = current_time + torch.distributions.Exponential(self.poisson_rate).sample().item()
                        else:
                            current_time = next_sell_time
                            arrived_sell = True
                            process_sell_order(volume=min(5000, self.V_max))
                            next_sell_time = current_time + torch.distributions.Exponential(self.poisson_rate).sample().item()
                    # Now one side arrived; the other is still missing
                    missing_buy = not arrived_buy
                    missing_sell = not arrived_sell
                # Now handle the remaining one missing side (if any), waiting for its first arrival
                if missing_buy and arrived_sell:
                    # Wait for first buy arrival after target
                    if next_buy_time <= self.tau_op - 1:
                        current_time = next_buy_time
                        arrived_buy = True
                        process_buy_order(volume=min(5000, self.V_max))
                        next_buy_time = current_time + torch.distributions.Exponential(self.poisson_rate).sample().item()
                elif missing_sell and arrived_buy:
                    # Wait for first sell arrival after target
                    if next_sell_time <= self.tau_op - 1:
                        current_time = next_sell_time
                        arrived_sell = True
                        process_sell_order(volume=min(5000, self.V_max))
                        next_sell_time = current_time + torch.distributions.Exponential(self.poisson_rate).sample().item()
                # The next decision time is the time of the last arrival (or tau_op-1 if no arrival occurred by then)
                next_decision_time = min(current_time, self.tau_op - 1)
            
            # Update order book depths after processing events
            self.depth_ask = next((j+1 for j,v in enumerate(self.ask_volumes) if v <= 1e-6), self.Lc)
            self.depth_bid = next((j+1 for j,v in enumerate(self.bid_volumes) if v <= 1e-6), self.Lc)
            # Update hypothetical clearing price H_t^cl (smoothly toward mid-price)
            tilde_price = self.mid_price
            self.H_cl += self.gamma * (tilde_price - self.H_cl)
            # Compute reward for this step
            S_submit = agent_price  # agent's limit price
            E_t = executed_vol       # executed volume of agent's order
            reward = S_submit * E_t * f(1 - self.kappa * f(self.H_cl - S_submit))
            # Update state variables
            self.last_executed = E_t
            self.inventory = max(0.0, self.inventory)  # inventory cannot go negative
            # Advance current time to the next decision time
            self.current_time = float(next_decision_time)
            self.last_action_time = float(next_decision_time)
            # Remove any remaining agent order (it will be replaced or canceled at next action)
            if self.agent_active_order_cont:
                self.agent_active_order_cont = None
            # Update mid-price via Brownian motion for the time interval
            dt = max(0.0, self.current_time - last_time)
            if dt > 0:
                price_change = torch.randn(1).item() * self.sigma_mid * math.sqrt(dt)
                self.mid_price += price_change
            # Check for transition to auction phase
            done = False
            if self.current_time >= self.tau_op - 1:
                # If we've reached the last continuous time, transition to auction
                self.phase = 'auction'
                # Start auction at time tau_op
                self.current_time = float(self.tau_op)
                # Reset auction-specific state
                self.N_plus = 0; self.N_minus = 0
                self.market_sell_volumes = [0.0] * self.L_max
                self.market_buy_volumes = [0.0] * self.L_max
                self.active_supply = []
            # Return next state
            next_state = self._get_state()
            return next_state, reward, done
        
        # Auction phase step
        elif self.phase == 'auction':
            # Expect action = (K_t^a, S_t^a, c_t) in the auction
            K_a, S_a, c_t = action
            K_a = float(K_a); S_a = float(S_a)
            # Current auction time (integer)
            t = int(self.current_time)
            done = False
            # 1. Process exogenous arrivals/cancellations at time t
            # (a) Exogenous limit (supply) orders
            # Randomly add a new supply order with some probability
            if torch.rand(1).item() < 0.3 and len(self.active_supply) < self.La:
                K_new = torch.rand(1).item() * (1 - 0.1) * 50  # e.g. uniform [0,50] as a slope
                S_new = self.mid_price + (torch.rand(1).item() - 0.5) * 10  # anchor around mid
                self.active_supply.append((K_new, S_new))
            # Randomly cancel an existing exogenous supply order
            if torch.rand(1).item() < 0.2 and self.active_supply:
                idx = int(torch.randint(0, len(self.active_supply), ()).item())
                self.active_supply.pop(idx)
            # (b) Exogenous market orders
            # New market sell order arrival
            if torch.rand(1).item() < 0.3:
                U = torch.rand(1).item()
                vol = self.v_m / ((1 - U) ** (1 / self.pareto_shape))
                vol = float(min(vol, self.V_max))
                if self.N_plus < self.L_max:
                    self.market_sell_volumes[self.N_plus] = vol
                self.N_plus += 1
                self.N_plus = min(self.N_plus, self.L_max)
            # New market buy order arrival
            if torch.rand(1).item() < 0.3:
                U = torch.rand(1).item()
                vol = self.v_m / ((1 - U) ** (1 / self.pareto_shape))
                vol = float(min(vol, self.V_max))
                if self.N_minus < self.L_max:
                    self.market_buy_volumes[self.N_minus] = vol
                self.N_minus += 1
                self.N_minus = min(self.N_minus, self.L_max)
            # Random cancellation of a pending market order
            if torch.rand(1).item() < 0.1:
                if self.N_plus > 0 and torch.rand(1).item() < 0.5:
                    # cancel a random sell MO
                    idx = int(torch.randint(0, self.N_plus, ()).item())
                    self.market_sell_volumes[idx] = 0.0
                if self.N_minus > 0 and torch.rand(1).item() < 0.5:
                    # cancel a random buy MO
                    idx = int(torch.randint(0, self.N_minus, ()).item())
                    self.market_buy_volumes[idx] = 0.0
            # 2. Apply agent's cancellations c_t
            if isinstance(c_t, torch.Tensor):
                c_t = c_t.tolist()
            # c_t is a binary vector length tau_cl (or tau_cl+1) indicating cancellations
            for s in range(self.tau_op, t):
                if s < len(c_t) and c_t[s] == 1 and self.agent_order_active[s]:
                    self.agent_order_active[s] = False
            # 3. Agent submits new order (K_t^a, S_t^a) if K_t^a > 0
            if K_a > 0:
                self.agent_orders_K[t] = K_a
                self.agent_orders_S[t] = S_a
                self.agent_order_active[t] = True
            # 4. Update hypothetical clearing price H_t^cl at time t
            # Compute total supply slope and intercept from active orders
            total_slope = 0.0
            total_intercept_term = 0.0
            # Sum exogenous supply functions
            for (K_i, S_i) in self.active_supply:
                total_slope += K_i
                total_intercept_term += K_i * S_i
            # Sum agent's active orders
            for s in range(self.tau_op, t+1):
                if self.agent_order_active[s]:
                    total_slope += self.agent_orders_K[s]
                    total_intercept_term += self.agent_orders_K[s] * self.agent_orders_S[s]
            # Sum net market order volume (sell orders contribute supply, buy orders demand)
            net_order_volume = 0.0
            if self.N_plus > 0:
                net_order_volume += sum(self.market_sell_volumes[:self.N_plus])
            if self.N_minus > 0:
                net_order_volume -= sum(self.market_buy_volumes[:self.N_minus])
            # Solve for clearing price p where total_slope * p - total_intercept_term + net_order_volume = 0
            if total_slope > 0:
                self.H_cl = (total_intercept_term - net_order_volume) / total_slope
            else:
                # No supply (should not happen if agent or others have orders); default to last price
                self.H_cl = self.mid_price
            # 5. Compute immediate reward at time t
            reward = 0.0
            # Market-making revenue component
            reward += K_a * (self.H_cl - S_a)
            # Penalty for wrong-side (buy-side) volume
            reward -= self.q * f(- K_a * (self.H_cl - S_a))
            # Penalty for cancellations (cost d per canceled order this step)
            cancel_count = 0
            if isinstance(c_t, (list, tuple)):
                cancel_count = sum(1 for s in range(self.tau_op, t) if s < len(c_t) and c_t[s] == 1)
            reward -= self.d * cancel_count
            # 6. Advance time
            prev_time = self.current_time
            self.last_executed = 0.0  # no volume executed immediately in auction
            self.current_time = float(t + 1)
            # 7. If we have reached the final time, clear the auction
            if self.current_time == float(self.tau_cl):
                # Compute final clearing price S_{tau_cl}^cl with all remaining orders
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
                    net_order_volume += sum(self.market_sell_volumes[:self.N_plus])
                if self.N_minus > 0:
                    net_order_volume -= sum(self.market_buy_volumes[:self.N_minus])
                if total_slope > 0:
                    clearing_price = (total_intercept_term - net_order_volume) / total_slope
                else:
                    clearing_price = self.H_cl
                # Calculate executed volume for the agent at final clearing
                executed_final = 0.0
                for s in range(self.tau_op, self.tau_cl):
                    if self.agent_order_active[s]:
                        executed_final += self.agent_orders_K[s] * (clearing_price - self.agent_orders_S[s])
                # Update inventory after auction
                I_final = self.inventory - executed_final
                self.inventory = I_final
                # Compute terminal reward components
                terminal_reward = executed_final - self.lambda_param * (abs(I_final) ** 2)
                wrong_side_sum = 0.0
                for s in range(self.tau_op, self.tau_cl):
                    if self.agent_order_active[s]:
                        # f(-K * (S_cl - S_s^a))
                        wrong_side_sum += f(- self.agent_orders_K[s] * (clearing_price - self.agent_orders_S[s]))
                terminal_reward -= self.q * wrong_side_sum
                reward += terminal_reward
                done = True
            # Return next state
            next_state = self._get_state()
            return next_state, reward, done
        
        # If phase is done (after terminal), do nothing
        else:
            return self._get_state(), 0.0, True
