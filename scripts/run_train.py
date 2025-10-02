# rl_auctions/scripts/run_train.py
import numpy as np
import torch
from config import EnvConfig, DQNConfig, TrainConfig
from env import MarketEmulator, StateEncoder, ActionSpace
from benchmark import GLFTBenchmark
from dqn.trainer import Trainer

def main():
    # 1) configs (fill with your paper-based params)
    env_cfg = EnvConfig(
        tau_open=100, tau_close=120, alpha=0.01, beta=0.05, B=10, V_max=5,
        L_c=10, L_a=10, L_takers=100, I_max=100,
        lambda_arrival=2.0, v_market=1, sigma_mid=0.2,
        pareto_vm=1.0, pareto_gamma=1.5, K_l=0.1, K_u=2.0, eps_contract=0.1,
        M_side_grid=5, p_new_mm=0.3, p_cancel_mm=0.1,
        p_new_taker=0.5, p_cancel_taker=0.1, gamma_smooth=0.5,
        kappa=1.0, q_wrong_side=1.0, d_cancel=0.01, lambda_term=0.1, initial_mid=100.0
    )
    encoder = StateEncoder(env_cfg.tau_close, env_cfg.alpha, env_cfg.B, env_cfg.L_c, env_cfg.L_a)
    actions = ActionSpace(env_cfg.V_max, env_cfg.B, env_cfg.beta, K_max=10, h=env_cfg.tau_close-env_cfg.tau_open)

    # 2) env & benchmark
    env = MarketEmulator(env_cfg, rng=np.random.default_rng(42))
    glft = GLFTBenchmark(A=1.0, k=0.1, sigma=env_cfg.sigma_mid, Q=env_cfg.I_max,
                     T=env_cfg.tau_open - 1, alpha_tick=env_cfg.alpha, gamma=0.0)

    # 3) dqn configs
    dqn_cfg = DQNConfig(obs_dim=encoder.obs_dim(), n_actions=actions.size())
    train_cfg = TrainConfig(episodes=500)

    # 4) trainer
    trainer = Trainer(env, encoder, actions, dqn_cfg, train_cfg, glft=glft, device="cpu")
    trainer.run()

if __name__ == "__main__":
    main()
