from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, List, Optional

import numpy as np
import torch

# Core package imports
from config import EnvConfig
from env import MarketEmulator, StateEncoder, ActionSpace
from env.gym_wrapper import make_gym_env  # optional; not strictly needed
from benchmark import GLFTBenchmark
from dqn.network import DQN
from dqn.agent import DQNAgent, EpsilonScheduler
from eval import Plotter

def build_default_env() -> tuple[MarketEmulator, StateEncoder, ActionSpace, GLFTBenchmark]:
    """
    Create a default emulator + encoder + action space + GLFT benchmark
    that match the config shown in run_train.py. Adjust as needed.
    """
    env_cfg = EnvConfig(
        tau_open=100, tau_close=120, alpha=0.01, beta=0.05, B=10, V_max=5,
        L_c=10, L_a=10, L_takers=100, I_max=100,
        lambda_arrival=10.0, v_market=5.0, sigma_mid=0.2,
        pareto_vm=5.0, pareto_gamma=1.5, K_l=0.1, K_u=2.0, eps_contract=0.1,
        M_side_grid=5, p_new_mm=0.3, p_cancel_mm=0.1,
        p_new_taker=0.5, p_cancel_taker=0.1, gamma_smooth=0.5,
        kappa=1.0, q_wrong_side=1.0, d_cancel=0.01, lambda_term=0.005, initial_mid=100.0,
    )

    encoder = StateEncoder(env_cfg.tau_close, env_cfg.alpha, env_cfg.B, env_cfg.L_c, env_cfg.L_a)
    encoder.set_horizon(env_cfg.tau_open, env_cfg.tau_close)
    actions = ActionSpace(env_cfg.V_max, env_cfg.B, env_cfg.beta, K_max=10, h=env_cfg.tau_close - env_cfg.tau_open,
                          alpha=env_cfg.alpha, S_anchor=100.0)

    env = MarketEmulator(env_cfg, rng=np.random.default_rng(42))
    glft = GLFTBenchmark(
        A=1.0,
        k=1.0,
        sigma=env_cfg.sigma_mid,
        Q=env_cfg.I_max,
        T=env_cfg.tau_open - 1,
        alpha_tick=env_cfg.alpha,
        gamma=0.0,  # risk-neutral limit as in our setup
    )
    return env, encoder, actions, glft


def load_q_from_checkpoint(q_net: DQN, target_net: DQN, ckpt_path: Optional[str]) -> bool:
    """
    Try to load a DQN checkpoint with flexible formats:
      - {'q_state_dict': ..., 'target_state_dict': ...}
      - {'state_dict': ...} (for q only)
      - state_dict directly
    Returns True if something was loaded successfully.
    """
    if not ckpt_path:
        print("[run_eval] No checkpoint provided; evaluating untrained network.")
        return False

    if not os.path.isfile(ckpt_path):
        print(f"[run_eval] Checkpoint not found: {ckpt_path}  (skipping load)")
        return False

    try:
        blob = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"[run_eval] Failed to load checkpoint: {e}  (skipping load)")
        return False

    loaded = False
    try:
        if isinstance(blob, dict) and "q_state_dict" in blob:
            q_net.load_state_dict(blob["q_state_dict"])
            if "target_state_dict" in blob:
                target_net.load_state_dict(blob["target_state_dict"])
            else:
                target_net.load_state_dict(blob["q_state_dict"])
            loaded = True
        elif isinstance(blob, dict) and "state_dict" in blob:
            q_net.load_state_dict(blob["state_dict"])
            target_net.load_state_dict(blob["state_dict"])
            loaded = True
        elif isinstance(blob, dict):
            # Heuristic: maybe it's directly a state_dict
            q_net.load_state_dict(blob)
            target_net.load_state_dict(blob)
            loaded = True
    except Exception as e:
        print(f"[run_eval] Checkpoint keys not compatible with network: {e}")

    if loaded:
        print(f"[run_eval] Loaded checkpoint from {ckpt_path}")
    else:
        print("[run_eval] Did not recognize checkpoint format; proceeding without weights.")
    return loaded


def greedy_eval(
    env: MarketEmulator,
    encoder: StateEncoder,
    actions: ActionSpace,
    q_net: DQN,
    episodes: int,
    glft: Optional[GLFTBenchmark] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run greedy (ε=0) evaluation episodes using the given q_net.
    """
    q_net = q_net.to(device)
    q_net.eval()

    # Make a dummy agent wrapper to use its ε=0 action method
    target = DQN(
        obs_dim=q_net.obs_dim,
        n_actions=q_net.n_actions,
        hidden_sizes=q_net.hidden_sizes,
        dueling=getattr(q_net, "dueling", False),
    ).to(device)
    target.eval()

    agent = DQNAgent(
        q_net=q_net,
        target_net=target,
        n_actions=actions.size(),
        lr=1e-3,
        tau_target=0.0,
        device=device,
    )
    eps0 = EpsilonScheduler(start=0.0, end=0.0, decay_steps=1)

    returns: List[float] = []
    regrets: List[float] = []

    for ep in range(max(1, int(episodes))):
        s = env.reset(seed=1234 + ep)
        obs = encoder.encode(s)
        done = False
        total_r = 0.0

        # GLFT starting value (regret reference)
        v_star_0 = None
        if glft is not None:
            try:
                q0 = int(env.cfg.I_max)
                s0 = float(s.get("mid", env.cfg.initial_mid))
                v_star_0 = float(glft.expected_value(x0=0.0, q0=q0, s0=s0))
            except Exception:
                v_star_0 = None

        steps = 0
        max_steps = int(env.cfg.tau_close)
        while not done and steps < max_steps:
            # Admissibility/mask
            adm = actions.admissible(
                t=s["t"],
                obs=s,
                constraints={
                    "inventory": s.get("inventory", 0),
                    "tau_open": env.cfg.tau_open,
                    "tau_close": env.cfg.tau_close,
                    "phase": s.get("phase"),
                },
            )
            a_idx = agent.act(obs=obs, step=0, eps_sched=eps0, admissible=adm)
            act_struct = actions.decode(a_idx)

            step_res = env.step(act_struct)
            total_r += float(step_res.reward)

            s = step_res.next_state
            obs = encoder.encode(s)
            done = bool(step_res.done)
            steps += 1

        returns.append(total_r)
        if v_star_0 is not None:
            regrets.append(float(v_star_0 - total_r))

    out = {
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "return_std": float(np.std(returns)) if returns else 0.0,
    }
    if regrets:
        out["regret_mean"] = float(np.mean(regrets))
        out["regret_std"] = float(np.std(regrets))
        out["regrets"] = regrets
    return out


def record_one_episode_trace(
    env: MarketEmulator,
    encoder: StateEncoder,
    actions: ActionSpace,
    q_net: DQN,
    device: str = "cpu",
) -> Dict[str, List[float]]:
    """
    Run a single greedy episode and record time-series for plotting.
    """
    q_net = q_net.to(device)
    q_net.eval()
    target = DQN(
        obs_dim=q_net.obs_dim,
        n_actions=q_net.n_actions,
        hidden_sizes=q_net.hidden_sizes,
        dueling=getattr(q_net, "dueling", False),
    ).to(device)
    target.eval()

    agent = DQNAgent(
        q_net=q_net,
        target_net=target,
        n_actions=actions.size(),
        lr=1e-3,
        tau_target=0.0,
        device=device,
    )
    eps0 = EpsilonScheduler(start=0.0, end=0.0, decay_steps=1)

    s = env.reset(seed=999)
    obs = encoder.encode(s)
    done = False

    traces = {"mid": [], "H_cl": [], "inventory": [], "reward": [], "S_star": []}
    S_cl_final = None

    steps = 0
    while not done and steps < env.cfg.tau_close:
        traces["mid"].append(float(s.get("mid", 0.0)))
        traces["H_cl"].append(float(s.get("H_cl", 0.0)))
        traces["inventory"].append(float(s.get("inventory", 0.0)))

        adm = actions.admissible(
            t=s["t"],
            obs=s,
            constraints={
                "inventory": s.get("inventory", 0),
                "tau_open": env.cfg.tau_open,
                "tau_close": env.cfg.tau_close,
                "phase": s.get("phase"),
            },
        )
        a_idx = agent.act(obs=obs, step=0, eps_sched=eps0, admissible=adm)
        act_struct = actions.decode(a_idx)
        step_res = env.step(act_struct)

        traces["reward"].append(float(step_res.reward))
        # CLOB executed price (if present in info)
        if "S_star" in step_res.info:
            traces["S_star"].append(float(step_res.info["S_star"]))
        else:
            traces["S_star"].append(np.nan)

        if "S_cl" in step_res.info:
            S_cl_final = float(step_res.info["S_cl"])

        s = step_res.next_state
        obs = encoder.encode(s)
        done = bool(step_res.done)
        steps += 1

    if S_cl_final is not None:
        traces["S_cl"] = S_cl_final
    return traces


def main():
    """Load checkpoint, run eval episodes, produce plots/tables."""
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN on the RL Auctions environment.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (.pt/.pth).")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--outdir", type=str, default="runs/eval", help="Directory to save plots and summary.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Torch device.")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Build environment stack
    env, encoder, actions, glft = build_default_env()

    # Build Q-network
    obs_dim = encoder.obs_dim()
    n_actions = actions.size()
    q = DQN(obs_dim=obs_dim, n_actions=n_actions, hidden_sizes=(256, 256))
    target = DQN(obs_dim=obs_dim, n_actions=n_actions, hidden_sizes=(256, 256))

    # Load weights (optional)
    load_q_from_checkpoint(q, target, args.ckpt)

    # Evaluation
    results = greedy_eval(env, encoder, actions, q, episodes=args.episodes, glft=glft, device=args.device)
    print("\n=== Evaluation Summary ===")
    for k in ["return_mean", "return_std", "regret_mean", "regret_std"]:
        if k in results:
            print(f"{k}: {results[k]:.6f}")

    # Save JSON summary
    summary_path = os.path.join(args.outdir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[run_eval] Wrote summary to {summary_path}")

    # Plots
    if not args.no_plots:
        p = Plotter()
        if "regrets" in results:
            p.plot_regret(results["regrets"])
            p.savefig(os.path.join(args.outdir, "regret.png"))

        traces = record_one_episode_trace(env, encoder, actions, q, device=args.device)
        p.plot_episode(traces)
        p.savefig(os.path.join(args.outdir, "episode.png"))
        print(f"[run_eval] Wrote plots to {args.outdir}")


if __name__ == "__main__":
    main()
