from __future__ import annotations

from typing import Dict, Any, Optional, List
import numpy as np
import torch

from env.market_emulator import MarketEmulator
from env.state_encoder import StateEncoder
from env.action_space import ActionSpace
from benchmark.glft import GLFTBenchmark
from eval import Logger, RegretAnalyzer, Plotter

from config import DQNConfig, TrainConfig
from .network import DQN
from .agent import DQNAgent, EpsilonScheduler
from .replay import ReplayBuffer

class Trainer:
    """
    DQN training driver:
      - Interacts with MarketEmulator using StateEncoder/ActionSpace
      - Learns a Q-network via uniform replay
      - Periodically evaluates vs GLFT benchmark to report regret
    """

    def __init__(
        self,
        env: MarketEmulator,
        encoder: StateEncoder,
        actions: ActionSpace,
        dqn_cfg: DQNConfig,
        train_cfg: TrainConfig,
        glft: Optional[GLFTBenchmark] = None,
        device: str = "cpu",
    ):
        self.env = env
        self.encoder = encoder
        self.actions = actions
        self.dqn_cfg = dqn_cfg
        self.train_cfg = train_cfg
        self.glft = glft
        self.device = device

        # Let the encoder know the episode horizon (for normalized time features)
        if hasattr(self.encoder, "set_horizon"):
            try:
                self.encoder.set_horizon(self.env.cfg.tau_open, self.env.cfg.tau_close)
            except Exception:
                pass

        # Build Q-networks and agent
        self.q = DQN(
            obs_dim=dqn_cfg.obs_dim,
            n_actions=dqn_cfg.n_actions,
            hidden_sizes=dqn_cfg.hidden_sizes,
        ).to(device)

        self.target = DQN(
            obs_dim=dqn_cfg.obs_dim,
            n_actions=dqn_cfg.n_actions,
            hidden_sizes=dqn_cfg.hidden_sizes,
        ).to(device)

        self.agent = DQNAgent(
            q_net=self.q,
            target_net=self.target,
            n_actions=dqn_cfg.n_actions,
            lr=dqn_cfg.lr,
            tau_target=dqn_cfg.tau_target,
            device=device,
        )

        # Replay buffer & epsilon schedule
        seed = dqn_cfg.seed if dqn_cfg.seed is not None else 42
        self.replay = ReplayBuffer(capacity=dqn_cfg.buffer_size, seed=seed)
        self.eps_sched = EpsilonScheduler(
            start=dqn_cfg.eps_start,
            end=dqn_cfg.eps_end,
            decay_steps=dqn_cfg.eps_decay_steps,
        )

        # Bookkeeping
        self.global_step: int = 0
        self.logger = Logger() if 'Logger' in globals() else None
        self.regret_analyzer = RegretAnalyzer() if 'RegretAnalyzer' in globals() else None
        self.plotter = Plotter() if 'Plotter' in globals() else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Main training loop across episodes with periodic evaluation."""
        episodes = int(self.train_cfg.episodes)
        for ep in range(1, episodes + 1):
            stats = self._collect_episode()
            # Basic console print (you can swap with a real logger)
            if (ep % self.train_cfg.log_interval) == 0 or ep == 1:
                print(
                    f"[Episode {ep:4d}] return={stats['return']:.4f} "
                    f"len={stats['length']} inv_T={stats['terminal_inventory']} "
                    f"avg_loss={stats.get('avg_loss','-')}"
                )

            # Periodic evaluation (greedy policy)
            if self.train_cfg.eval_every and (ep % self.train_cfg.eval_every) == 0:
                eval_res = self._evaluate(episodes=self.train_cfg.eval_episodes)
                print(
                    f"  -> Eval: return_mean={eval_res['return_mean']:.4f} "
                    f"regret_mean={eval_res.get('regret_mean','-')}"
                )

    # ------------------------------------------------------------------
    # Episode collection & learning
    # ------------------------------------------------------------------
    def _collect_episode(self) -> Dict[str, Any]:
        """Roll out a single episode into replay; return stats."""
        # Reset env
        s_dict = self.env.reset()
        obs = self.encoder.encode(s_dict)

        done = False
        total_reward = 0.0
        steps = 0
        losses: List[float] = []

        # For regret proxy: benchmark value at episode start (optional)
        v_star_0 = None
        if self.glft is not None:
            try:
                q0 = int(self.env.cfg.I_max)
                s0 = float(s_dict.get("mid", self.env.cfg.initial_mid))
                x0 = 0.0
                v_star_0 = float(self.glft.expected_value(x0=x0, q0=q0, s0=s0))
            except Exception:
                v_star_0 = None

        # Max steps (defaults to τ_cl)
        max_steps = (
            int(self.train_cfg.max_steps_per_episode)
            if self.train_cfg.max_steps_per_episode is not None
            else int(self.env.cfg.tau_close)
        )

        while not done and steps < max_steps:
            # Admissible actions (phase-aware)
            adm = self.actions.admissible(
                t=s_dict["t"],
                obs=s_dict,
                constraints={
                    "inventory": s_dict.get("inventory", 0),
                    "tau_open": self.env.cfg.tau_open,
                    "tau_close": self.env.cfg.tau_close,
                    "phase": s_dict.get("phase"),
                },
            )

            # ε-greedy action
            a_idx = self.agent.act(
                obs=obs,
                step=self.global_step,
                eps_sched=self.eps_sched,
                admissible=adm,
            )

            # Structured action for env
            act_struct = self.actions.decode(a_idx)

            # Step env
            step_res = self.env.step(act_struct)
            s_next = step_res.next_state
            r = float(step_res.reward)
            done = bool(step_res.done)

            obs_next = self.encoder.encode(s_next)

            # Push to replay
            self.replay.push(
                t=s_dict["t"],
                obs=obs,
                a=a_idx,
                r=r,
                obs_next=obs_next,
                done=done,
            )

            # Learning
            if len(self.replay) >= self.dqn_cfg.train_start and (self.global_step % self.dqn_cfg.train_freq == 0):
                try:
                    batch = self.replay.sample(self.dqn_cfg.batch_size)
                    stats = self.agent.learn(batch, gamma=self.dqn_cfg.gamma)
                    losses.append(stats["loss"])
                except ValueError:
                    # not enough samples; skip
                    pass

            if self.global_step % self.dqn_cfg.target_update_freq == 0:
                self.agent.update_target()

            # Accumulate
            total_reward += r
            steps += 1
            self.global_step += 1

            # Advance state
            s_dict = s_next
            obs = obs_next

        # End-of-episode stats
        terminal_inventory = int(s_dict.get("inventory", 0))
        avg_loss = float(np.mean(losses)) if losses else None

        # Regret proxy (if GLFT provided)
        if v_star_0 is not None:
            v_pi_0 = float(total_reward)
            regret = float(v_star_0 - v_pi_0)
            if self.regret_analyzer is not None:
                try:
                    self.regret_analyzer.add_episode(v_star_0=v_star_0, v_pi_0=v_pi_0)
                except Exception:
                    pass
        else:
            regret = None

        # Optional logging hook
        if self.logger is not None:
            try:
                self.logger.log(
                    step=self.global_step,
                    scalars={
                        "episode_return": total_reward,
                        "episode_length": steps,
                        "terminal_inventory": terminal_inventory,
                        **({"avg_loss": avg_loss} if avg_loss is not None else {}),
                        **({"regret": regret} if regret is not None else {}),
                    },
                )
            except Exception:
                pass

        return {
            "return": float(total_reward),
            "length": int(steps),
            "terminal_inventory": terminal_inventory,
            **({"avg_loss": float(avg_loss)} if avg_loss is not None else {}),
            **({"regret": float(regret)} if regret is not None else {}),
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate(self, episodes: int = 10) -> Dict[str, float]:
        """
        Run greedy episodes (ε=0) and compute mean return.
        If a GLFT benchmark is provided, also compute a regret proxy:
            regret_e = V*_0(e) - R_e
        """
        returns: List[float] = []
        regrets: List[float] = []

        greedy_eps = EpsilonScheduler(start=0.0, end=0.0, decay_steps=1)

        for _ in range(max(1, int(episodes))):
            s_dict = self.env.reset()
            obs = self.encoder.encode(s_dict)
            done = False
            total_reward = 0.0
            # Benchmark start value
            v_star_0 = None
            if self.glft is not None:
                try:
                    q0 = int(self.env.cfg.I_max)
                    s0 = float(s_dict.get("mid", self.env.cfg.initial_mid))
                    v_star_0 = float(self.glft.expected_value(x0=0.0, q0=q0, s0=s0))
                except Exception:
                    v_star_0 = None

            steps = 0
            max_steps = int(self.env.cfg.tau_close)
            while not done and steps < max_steps:
                adm = self.actions.admissible(
                    t=s_dict["t"],
                    obs=s_dict,
                    constraints={
                        "inventory": s_dict.get("inventory", 0),
                        "tau_open": self.env.cfg.tau_open,
                        "tau_close": self.env.cfg.tau_close,
                        "phase": s_dict.get("phase"),
                    },
                )
                a_idx = self.agent.act(obs=obs, step=0, eps_sched=greedy_eps, admissible=adm)
                act_struct = self.actions.decode(a_idx)
                step_res = self.env.step(act_struct)
                total_reward += float(step_res.reward)
                s_dict = step_res.next_state
                obs = self.encoder.encode(s_dict)
                done = bool(step_res.done)
                steps += 1

            returns.append(total_reward)
            if v_star_0 is not None:
                regrets.append(float(v_star_0 - total_reward))

        ret_mean = float(np.mean(returns)) if returns else 0.0
        reg_mean = float(np.mean(regrets)) if regrets else None

        # Optional logging
        if self.logger is not None:
            try:
                scal = {"eval_return_mean": ret_mean}
                if reg_mean is not None:
                    scal["eval_regret_mean"] = reg_mean
                self.logger.log(step=self.global_step, scalars=scal)
            except Exception:
                pass

        return {"return_mean": ret_mean, **({"regret_mean": reg_mean} if reg_mean is not None else {})}
