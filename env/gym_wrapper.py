# rl_auctions/env/gym_wrapper.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise ImportError(
        "This wrapper requires `gymnasium`. Install via `pip install gymnasium`."
    ) from e

from .market_emulator import MarketEmulator, EnvConfig

class AuctionMarketGym(gym.Env):
    """
    Gymnasium-style wrapper around MarketEmulator.

    Observations:
        Flat float32 vector produced by your StateEncoder.encode(state_dict).

    Actions:
        Discrete integer a∈{0..N-1}. We decode to structured actions via your ActionSpace.

    Info dict:
        - "phase": "clob" | "auction" | "terminal"
        - "t": current step
        - "raw_state": (optional) minimal raw state (mid, H_cl, inventory, etc.)
        - "action_mask": np.ndarray shape (n_actions,), dtype=bool (True = admissible)
        - plus pass-through emulator info (e.g., S_star, H_cl, executions...)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        emulator: MarketEmulator,
        encoder,           # StateEncoder (must implement obs_dim() and encode(state_dict)->np.ndarray)
        actions,           # ActionSpace (must implement size(), decode(a_idx)->dict, admissible(...)->List[int])
        max_episode_steps: Optional[int] = None,
        return_raw_state_in_info: bool = True,
    ):
        super().__init__()
        self.emu = emulator
        self.encoder = encoder
        self.actions = actions
        self.return_raw = bool(return_raw_state_in_info)

        # Spaces
        obs_dim = int(self.encoder.obs_dim())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(int(self.actions.size()))

        # Episode control
        self._max_steps = int(max_episode_steps) if max_episode_steps is not None else self.emu.cfg.tau_close
        self._step_count = 0
        self._last_obs: Optional[np.ndarray] = None

        # Seed
        self._seed: Optional[int] = None

    # --------------- Gymnasium API ---------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Returns:
            obs (np.ndarray, float32), info (dict)
        """
        if seed is not None:
            self._seed = int(seed)
        state = self.emu.reset(seed=self._seed)
        self._step_count = 0

        obs = self._encode(state)
        info = self._info(state)
        self._last_obs = obs
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Args:
            action: integer in [0, N-1]. Decoded by ActionSpace.decode.

        Returns:
            obs, reward, terminated, truncated, info
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        phase = self.emu.current_phase()

        # Structured action for emulator
        act_struct = self.actions.decode(int(action))

        # Step the environment
        step_res = self.emu.step(act_struct)
        state_next = step_res.next_state
        reward = float(step_res.reward)

        self._step_count += 1

        # Termination and truncation flags
        terminated = bool(step_res.done)  # episode ends when t == tau_close
        truncated = False
        if self._step_count >= self._max_steps and not terminated:
            truncated = True
            terminated = True  # gymnasium convention: end episode

        # Encode next observation
        obs = self._encode(state_next)
        info = self._info(state_next, extra=step_res.info)

        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Text render; extend as needed."""
        s = self.emu.get_state_dict()
        print(
            f"[t={s['t']:3d} | {s['phase']}] "
            f"mid={s['mid']:.4f} H_cl={s['H_cl']:.4f} inv={s['inventory']}"
        )

    def close(self) -> None:
        """Nothing to close (stateless wrapper)."""
        return

    # --------------- Helpers ---------------

    def _encode(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """Use provided encoder to produce float32 vector."""
        obs = self.encoder.encode(state_dict)
        obs = np.asarray(obs, dtype=np.float32)
        assert self.observation_space.contains(obs), "Encoder output not in observation_space."
        return obs

    def _info(self, state_dict: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compose the info dict, including an action mask if available."""
        info: Dict[str, Any] = {
            "phase": state_dict.get("phase"),
            "t": int(state_dict.get("t", -1)),
        }

        # Optional raw summary
        if self.return_raw:
            info["raw_state"] = {
                "mid": state_dict.get("mid"),
                "H_cl": state_dict.get("H_cl"),
                "inventory": state_dict.get("inventory"),
            }

        # Action mask using ActionSpace.admissible if provided
        try:
            adm = self.actions.admissible(
                t=state_dict.get("t", 0),
                obs=state_dict,
                constraints={
                    "inventory": state_dict.get("inventory"),
                    "tau_open": self.emu.cfg.tau_open,
                    "tau_close": self.emu.cfg.tau_close,
                    "phase": state_dict.get("phase"),
                },
            )
            mask = np.zeros(self.action_space.n, dtype=bool)
            mask[np.asarray(adm, dtype=int)] = True
            info["action_mask"] = mask
        except Exception:
            # If not implemented yet, skip silently
            pass

        if extra:
            info.update(extra)
        return info


# ---------------- Factory ----------------

def make_gym_env(
    cfg: EnvConfig,
    encoder,
    actions,
    rng: Optional[np.random.Generator] = None,
    max_episode_steps: Optional[int] = None,
    return_raw_state_in_info: bool = True,
) -> AuctionMarketGym:
    """
    Convenience constructor that builds the MarketEmulator and wraps it.

    Example:
        env = make_gym_env(cfg, encoder, actions)
        obs, info = env.reset(seed=42)
        obs, r, term, trunc, info = env.step(env.action_space.sample())
    """
    emu = MarketEmulator(cfg=cfg, rng=rng)
    return AuctionMarketGym(
        emulator=emu,
        encoder=encoder,
        actions=actions,
        max_episode_steps=max_episode_steps,
        return_raw_state_in_info=return_raw_state_in_info,
    )
