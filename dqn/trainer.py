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
        Run greedy episodes (ε=0) and compute mean return for the RL policy.
        Additionally, if a GLFT benchmark is provided, replay the SAME episodes
        with the GLFT policy (CLOB-only; no auction trading; leftover liquidated
        at S_T^mid where T=τ_op-1) to obtain comparable realized returns.
        We also keep the original value-regret proxy against V*_0 for continuity.
        """
        import random

        def _snapshot_seeds():
            seeds = {"py": random.getstate(), "np": np.random.get_state()}
            try:
                import torch as _torch
                seeds["torch"] = _torch.random.get_rng_state()
            except Exception:
                pass
            return seeds

        def _restore_seeds(seeds):
            random.setstate(seeds["py"])
            np.random.set_state(seeds["np"])
            try:
                import torch as _torch
                if "torch" in seeds:
                    _torch.random.set_rng_state(seeds["torch"])
            except Exception:
                pass

        def _extract_delta_and_volume(act):
            # Pull (delta_ticks, volume) from a decoded action (dict/object/tuple).
            delta = None
            vol = None
            if isinstance(act, dict):
                for k in ("delta_ticks", "delta", "ask_delta", "d"):
                    if k in act:
                        delta = float(act[k]); break
                for k in ("volume", "v", "qty", "size"):
                    if k in act:
                        vol = int(act[k]); break
            else:
                # tuple-like
                if hasattr(act, "__iter__") and not hasattr(act, "keys"):
                    lst = list(act)
                    if len(lst) >= 2:
                        try: vol = int(lst[0])
                        except Exception: pass
                        try: delta = float(lst[1])
                        except Exception: pass
                # attributes
                if delta is None:
                    for k in ("delta_ticks", "delta", "ask_delta", "d"):
                        if hasattr(act, k):
                            try:
                                delta = float(getattr(act, k)); break
                            except Exception:
                                pass
                if vol is None:
                    for k in ("volume", "v", "qty", "size"):
                        if hasattr(act, k):
                            try:
                                vol = int(getattr(act, k)); break
                            except Exception:
                                pass
            return delta, vol

        def _choose_idx_for_delta(delta_ticks_target: float, q: int, adm_indices: List[int]) -> int:
            # Among admissible indices, pick the one with vol<=q and delta closest to target.
            # If none satisfy vol<=q, ignore vol. Prefer larger volume on ties.
            best = None; best_key = (float("inf"), 0)      # (|Δ|, -vol)
            fallback = None; fallback_key = (float("inf"), 0)
            for idx in adm_indices:
                act = self.actions.decode(idx)
                d, v = _extract_delta_and_volume(act)
                if d is None:  # skip if no delta info
                    continue
                v_eff = int(v) if v is not None else 0
                key = (abs(float(d) - float(delta_ticks_target)), -v_eff)
                if v_eff <= max(0, int(q)):
                    if key < best_key: best_key, best = key, idx
                else:
                    if key < fallback_key: fallback_key, fallback = key, idx
            return best if best is not None else (fallback if fallback is not None else adm_indices[0])

        # -------- Pass 1: RL greedy episodes (store returns + per-episode seeds) --------
        rl_returns: List[float] = []
        regrets_value: List[float] = []
        episode_seeds: List[dict] = []

        greedy_eps = EpsilonScheduler(start=0.0, end=0.0, decay_steps=1)

        for _ in range(max(1, int(episodes))):
            episode_seeds.append(_snapshot_seeds())

            s_dict = self.env.reset()
            obs = self.encoder.encode(s_dict)
            done = False
            total_reward = 0.0
            steps = 0

            # Legacy value-regret baseline V*_0 (optional)
            v_star_0 = None
            if self.glft is not None:
                try:
                    q0 = int(self.env.cfg.I_max)
                    s0 = float(s_dict.get("mid", getattr(self.env.cfg, "initial_mid", 0.0)))
                    x0 = 0.0
                    v_star_0 = float(self.glft.expected_value(x0=x0, q0=q0, s0=s0))
                except Exception:
                    v_star_0 = None

            max_steps = (
                int(self.train_cfg.max_steps_per_episode)
                if self.train_cfg.max_steps_per_episode is not None
                else int(self.env.cfg.tau_close)
            )

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

            rl_returns.append(total_reward)
            if v_star_0 is not None:
                regrets_value.append(float(v_star_0 - total_reward))

        # -------- Pass 2: GLFT episodes on identical randomness (policy returns) --------
        glft_returns: List[float] = []
        if self.glft is not None:
            for seeds in episode_seeds:
                _restore_seeds(seeds)
                s = self.env.reset()
                done = False
                ret = 0.0
                prev_s = None

                while not done:
                    phase = s.get("phase", "C")

                    if phase in ("continuous", "C", "clob"):
                        q = int(s.get("inventory", 0))
                        # Admissible actions at this step
                        adm = self.actions.admissible(
                            t=s["t"],
                            obs=s,
                            constraints={
                                "inventory": q,
                                "tau_open": self.env.cfg.tau_open,
                                "tau_close": self.env.cfg.tau_close,
                                "phase": phase,
                            },
                        )

                        if q <= 0:
                            # Prefer explicit no-trade if present; else pick largest delta (lowest fill prob)
                            a_idx = None
                            for idx in adm:
                                act = self.actions.decode(idx)
                                _, v = _extract_delta_and_volume(act)
                                if v == 0:
                                    a_idx = idx; break
                            if a_idx is None:
                                max_d = -1e18; pick = adm[0]
                                for idx in adm:
                                    act = self.actions.decode(idx)
                                    d, _ = _extract_delta_and_volume(act)
                                    if d is not None and d > max_d:
                                        max_d, pick = d, idx
                                a_idx = pick
                        else:
                            # δ* (currency) → ticks, then snap to nearest admissible action
                            delta_cur = float(self.glft.delta_star(t=s["t"], q=q))
                            delta_ticks = delta_cur / float(self.glft.alpha_tick)
                            a_idx = _choose_idx_for_delta(delta_ticks_target=delta_ticks, q=q, adm_indices=adm)

                        act_struct = self.actions.decode(a_idx)
                        step = self.env.step(act_struct)
                        ret += float(step.reward)

                        prev_s = s
                        s = step.next_state
                        done = bool(step.done)

                        # Enforce CLOB-only: stop at auction boundary and liquidate leftovers at S_T^mid
                        phase_after = s.get("phase", phase)
                        crossed = (phase in ("continuous","C","clob")) and (phase_after in ("auction","A"))
                        ended_at_clob_end = done and (phase in ("continuous","C","clob"))
                        if crossed or ended_at_clob_end:
                            q_left = int(s.get("inventory", 0))
                            if q_left > 0:
                                S_mid_T = float((prev_s or s).get("mid", getattr(self.env.cfg, "initial_mid", 0.0)))
                                ret += float(q_left) * S_mid_T
                            break

                    elif phase in ("auction", "A"):
                        # Shouldn't happen—loop breaks at boundary above; ignore auction reward by design
                        break

                    else:
                        # Unknown phase: safe no-op step if the env supports it
                        step = self.env.step((0, 0, 0))
                        s = step.next_state
                        done = bool(step.done)

                glft_returns.append(float(ret))

        # -------- Aggregates & logging --------
        ret_mean = float(np.mean(rl_returns)) if rl_returns else 0.0
        reg_mean = float(np.mean(regrets_value)) if regrets_value else None
        glft_mean = float(np.mean(glft_returns)) if glft_returns else None
        gap_mean = (
            float(np.mean([g - r for g, r in zip(glft_returns, rl_returns)]))
            if glft_returns and rl_returns else None
        )

        if self.logger is not None:
            try:
                scal = {"eval_return_mean": ret_mean}
                if glft_mean is not None:
                    scal["eval_glft_return_mean"] = glft_mean
                if gap_mean is not None:
                    scal["eval_policy_gap_mean"] = gap_mean
                if reg_mean is not None:
                    scal["eval_regret_mean"] = reg_mean
                self.logger.log(step=self.global_step, scalars=scal)
            except Exception:
                pass

        out = {"return_mean": ret_mean}
        if glft_mean is not None:
            out["glft_return_mean"] = glft_mean
        if gap_mean is not None:
            out["policy_gap_mean"] = gap_mean
        if reg_mean is not None:
            out["regret_mean"] = reg_mean
        return out
