from __future__ import annotations

from typing import Deque, Tuple, Any, Dict, List
from collections import deque
import random
import numpy as np


class ReplayBuffer:
    """
    Uniform replay with transitions:
        (t, obs, action, reward, next_obs, done)

    Notes
    -----
    - Stores observations as float32 and rewards as float32 to save memory.
    - Sampling returns numpy arrays stacked/batched:
        {
          "t":        (B,),
          "obs":      (B, *obs_shape),
          "actions":  (B,),
          "rewards":  (B,),
          "next_obs": (B, *obs_shape),
          "dones":    (B,),
        }
    - The learner (agent) can then convert to torch tensors on its device.
    """

    def __init__(self, capacity: int, seed: int = 42):
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive.")
        self.capacity: int = int(capacity)
        self.buffer: Deque[Tuple[int, np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=self.capacity)
        self.rng = random.Random(seed)

        # Optional: cache observation shape after first push for validation
        self._obs_shape: Tuple[int, ...] | None = None

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        t: int,
        obs: np.ndarray,
        a: int,
        r: float,
        obs_next: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add one transition to the buffer.
        """
        # Ensure arrays/dtypes are compact
        obs_arr = np.asarray(obs, dtype=np.float32).copy()
        next_arr = np.asarray(obs_next, dtype=np.float32).copy()

        if self._obs_shape is None:
            self._obs_shape = tuple(obs_arr.shape)
        else:
            # Basic shape check; tolerate mismatches by reshaping if flat
            if tuple(obs_arr.shape) != self._obs_shape:
                try:
                    obs_arr = obs_arr.reshape(self._obs_shape)
                    next_arr = next_arr.reshape(self._obs_shape)
                except Exception as _:
                    pass  # leave as-is; learner may still handle it

        self.buffer.append(
            (int(t), obs_arr, int(a), float(r), next_arr, bool(done))
        )

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample a minibatch uniformly at random.

        Returns
        -------
        dict of numpy arrays:
            - "t": (B,)
            - "obs": (B, *obs_shape)
            - "actions": (B,)
            - "rewards": (B,)
            - "next_obs": (B, *obs_shape)
            - "dones": (B,)
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer: have {len(self.buffer)}, need {batch_size}.")

        batch = self.rng.sample(self.buffer, k=batch_size)

        t_list: List[int] = []
        obs_list: List[np.ndarray] = []
        act_list: List[int] = []
        rew_list: List[float] = []
        nxt_list: List[np.ndarray] = []
        done_list: List[bool] = []

        for (t, s, a, r, sp, d) in batch:
            t_list.append(t)
            obs_list.append(s)
            act_list.append(a)
            rew_list.append(r)
            nxt_list.append(sp)
            done_list.append(d)

        # Stack into arrays
        obs = np.stack(obs_list, axis=0).astype(np.float32, copy=False)
        next_obs = np.stack(nxt_list, axis=0).astype(np.float32, copy=False)
        t_arr = np.asarray(t_list, dtype=np.int64)
        actions = np.asarray(act_list, dtype=np.int64)
        rewards = np.asarray(rew_list, dtype=np.float32)
        dones = np.asarray(done_list, dtype=np.bool_)

        return {
            "t": t_arr,
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
        }
