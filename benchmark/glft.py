from __future__ import annotations

from typing import Dict, Tuple, Optional, List
import numpy as np


class GLFTBenchmark:
    """
    GLFT benchmark for the CLOB phase with liquidation by T=τ_op-1.
    Risk-neutral default (γ=0): α=0, η=A/e. Handles α>0 too via expmv series.

    API:
      - v_q(t): returns [v_0(t),...,v_Q(t)]
      - delta_star(t,q): optimal quote δ*(t,q)
      - policy(): map (t,q)->δ*
      - expected_value(x0,q0,s0): CE at t=0 in γ→0 limit
    """

    def __init__(
        self,
        A: float,
        k: float,
        sigma: float,
        Q: int,
        T: int,
        alpha_tick: float,
        gamma: float = 0.0,
        eps: float = 1e-12,
        max_series_terms: int = 512,
    ):
        self.A = float(A)
        self.k = float(k)
        self.sigma = float(sigma)
        self.Q = int(Q)
        self.T = int(T)
        self.alpha_tick = float(alpha_tick)
        self.gamma = float(max(0.0, gamma))
        self.eps = float(eps)
        self.max_series_terms = int(max_series_terms)

        # Coefficients
        if self.gamma == 0.0:
            self.alpha_coef = 0.0
            self.eta = self.A / np.e
        else:
            self.alpha_coef = (self.k * self.gamma * (self.sigma ** 2)) / 2.0
            self.eta = self.A * (1.0 + self.gamma / self.k) ** (-(1.0 + self.k / self.gamma))

        # Prebuild M structure (lower bidiagonal) and a fast matvec
        self.n = self.Q + 1
        # No need to materialize the full matrix; keep params and implement matvec.
        # But for clarity we store a dense copy once (small n).
        self.M = self._build_M(self.Q, self.alpha_coef, self.eta)

        # Constant vector e = ones(Q+1)
        self._e = np.ones(self.n, dtype=float)

        # Cache for v(t)
        self._v_cache: Dict[int, np.ndarray] = {}

    # ------------------- Public API -------------------

    def v_q(self, t: int) -> np.ndarray:
        """Return [v_0(t),...,v_Q(t)] via v(t) = exp(-M (T-t)) e."""
        t = int(t)
        if not (0 <= t <= self.T):
            raise ValueError(f"t must be in [0, {self.T}]")

        if t in self._v_cache:
            return self._v_cache[t]

        tau = float(self.T - t)
        v = self._expm_times_vec(tau, self._e)
        v = np.maximum(v, self.eps)  # guard
        self._v_cache[t] = v
        return v

    def delta_star(self, t: int, q: int, gamma: Optional[float] = None) -> float:
        """δ*(t,q) = (1/k) ln(v_q/v_{q-1}) + (1/γ)ln(1+γ/k); for γ=0: +1/k."""
        if q <= 0:
            return float("inf")
        if q > self.Q:
            q = self.Q

        v = self.v_q(t)
        num = max(float(v[q]), self.eps)
        den = max(float(v[q - 1]), self.eps)

        if gamma is None:
            gamma = self.gamma

        core = (1.0 / self.k) * np.log(num / den)
        if gamma == 0.0:
            offset = 1.0 / self.k
        else:
            offset = (1.0 / gamma) * np.log(1.0 + gamma / self.k)
        return float(core + offset)

    def policy(self) -> Dict[tuple[int, int], float]:
        """Map (t,q) -> δ*(t,q) for all t=0..T, q=0..Q."""
        out: Dict[tuple[int, int], float] = {}
        for t in range(self.T + 1):
            _ = self.v_q(t)  # warm cache
            for q in range(self.Q + 1):
                out[(t, q)] = self.delta_star(t, q)
        return out

    def expected_value(self, x0: float, q0: int, s0: float) -> float:
        """V*_0 = x0 + q0 s0 + (1/k) ln v_{q0}(0) in γ→0."""
        q0 = max(0, min(int(q0), self.Q))
        v0 = max(float(self.v_q(0)[q0]), self.eps)
        return float(x0 + q0 * s0 + (1.0 / self.k) * np.log(v0))

    # ------------------- Internals -------------------

    @staticmethod
    def _build_M(Q: int, alpha_coef: float, eta: float) -> np.ndarray:
        n = Q + 1
        M = np.zeros((n, n), dtype=float)
        for q in range(n):
            M[q, q] = alpha_coef * (q ** 2)
            if q >= 1:
                M[q, q - 1] = -eta
        return M

    def _matvec_M(self, x: np.ndarray) -> np.ndarray:
        """
        Fast M @ x for lower-bidiagonal M:
            y[q] = α q^2 x[q] - η x[q-1]  (with x[-1]=0)
        """
        y = np.zeros_like(x, dtype=float)
        # diagonal term
        if self.alpha_coef != 0.0:
            # q=0 diagonal is 0; start at q=1 to avoid mul by 0
            for q in range(self.n):
                y[q] += self.alpha_coef * (q ** 2) * x[q]
        # subdiagonal term
        for q in range(1, self.n):
            y[q] += -self.eta * x[q - 1]
        return y

    def _expm_times_vec(self, tau: float, vec: np.ndarray) -> np.ndarray:
        """
        Compute exp(-M τ) @ vec.
        - If α=0, M is nilpotent → finite series up to degree Q (exact).
        - Else, use power-series expmv with adaptive truncation.
        """
        if tau == 0.0:
            return vec.astype(float, copy=True)

        if self.alpha_coef == 0.0:
            # Exact finite series: sum_{n=0..Q} [(-τ)^n/n!] M^n vec
            out = vec.astype(float, copy=True)
            term = vec.astype(float, copy=True)  # M^0 @ vec
            coeff = 1.0
            for n in range(1, self.n):  # degree ≤ Q
                # Update term: term_n = M @ term_{n-1}
                term = self._matvec_M(term)
                coeff *= (-tau) / n
                out += coeff * term
            return out

        # General α>0: expmv series with adaptive stop
        out = vec.astype(float, copy=True)
        term = vec.astype(float, copy=True)
        coeff = 1.0
        for n in range(1, self.max_series_terms + 1):
            term = self._matvec_M(term)          # M^n @ vec via recurrence
            coeff *= (-tau) / n                  # (-τ)^n / n!
            incr = coeff * term
            out += incr
            # adaptive stop on 1-norm
            if np.linalg.norm(incr, 1) <= self.eps * (np.linalg.norm(out, 1) + 1.0):
                break
        return out


# glft.py  (add near the end of the file)

from typing import Sequence
import random
import numpy as np

def run_glft_benchmark_episodes(env, glft, episode_seeds: Sequence[dict], *, stop_at_tauop: bool = True):
    """
    Replays len(episode_seeds) episodes with identical randomness,
    using the GLFT policy on the CLOB and doing nothing in the auction.
    If stop_at_tauop=True, the episode is considered ended at T (CLOB end),
    leftover inventory is liquidated at S_T^mid, and auction rewards are ignored.

    Returns: list[float] of episode returns (cumulated reward + forced liquidation).
    """
    bm_returns = []

    def _normalize_step_out(out):
        if not isinstance(out, tuple):
            raise TypeError(f"env step returned non-tuple: {type(out)}")
        n = len(out)
        if n == 5:
            s, r, terminated, truncated, info = out
            return s, float(r), bool(terminated) or bool(truncated), info
        if n == 4:
            s, r, done, info = out
            return s, float(r), bool(done), info
        if n == 3:
            s, r, done = out
            return s, float(r), bool(done), {}
        raise ValueError(f"Unsupported step return length {n}")

    def _state_get_midprice(state):
        # add/adjust keys to match your env
        for k in ("mid", "mid_price", "S_mid", "mid_px", "price_mid"):
            try:
                if isinstance(state, dict) and k in state: return float(state[k])
                if hasattr(state, k): return float(getattr(state, k))
            except Exception:
                pass
        # fallback: env has a getter
        for name in ("get_mid", "mid", "mid_price"):
            fn = getattr(env, name, None)
            if callable(fn):
                try: return float(fn())
                except Exception: pass
        raise AttributeError("Cannot extract mid-price; add your key above.")

    def _state_get_inventory(state):
        for k in ("inv","inventory","I","q","qty","position"):
            try:
                if isinstance(state, dict) and k in state: return int(state[k])
                if hasattr(state, k): return int(getattr(state, k))
            except Exception:
                pass
        raise AttributeError("Cannot extract inventory; add your key above.")

    def _build_auction_noop(env):
        # (c_t, K_a, S_a) with zero vector c_t
        n = 0
        if hasattr(env, "agent_order_active") and env.agent_order_active is not None:
            n = len(env.agent_order_active)
        c_t = [0] * int(n)
        return (0.0, 0.0, c_t)

    for seeds in episode_seeds:
        # restore PRNGs to replay exactly the same episode noise
        random.setstate(seeds["py"])
        np.random.set_state(seeds["np"])
        try:
            import torch
            torch.random.set_rng_state(seeds["torch"])
        except Exception:
            pass

        s = env.reset()
        done = False
        t_idx = 0
        ret = 0.0
        prev_s = None

        while not done:
            phase = getattr(env, "phase", "C")

            if phase in ("continuous","C","clob"):
                q = _state_get_inventory(s)
                if q <= 0:
                    # explicit no-trade action if you have one, else min volume & large delta
                    action = (0.0, 0.0)  # adjust if your env expects an index
                else:
                    # choose delta* in TICKS and map to your (volume, delta_ticks) action here
                    delta_ticks = glft.delta_star_ticks(t_idx, q)
                    # example: (v=1, delta_ticks)
                    action = (1.0, float(delta_ticks))

                if hasattr(env, "step_clob"):
                    s_next, r, done, info = _normalize_step_out(env.step_clob(action))
                else:
                    s_next, r, done, info = _normalize_step_out(env.step(action))

                # Did we just leave the CLOB?
                phase_after = getattr(env, "phase", phase)
                crossed_to_auction = (phase in ("continuous","C","clob")) and (phase_after in ("auction","A"))
                ended_at_clob_end = done and (phase in ("continuous","C","clob"))

                ret += r
                t_idx += 1

                if crossed_to_auction or ended_at_clob_end:
                    # Force liquidation of any leftover at S_T^mid (T is the last CLOB index).
                    q_left = _state_get_inventory(s_next)
                    if q_left > 0:
                        S_mid_T = _state_get_midprice(prev_s if prev_s is not None else s_next)
                        ret += float(q_left) * float(S_mid_T)
                    if stop_at_tauop:
                        break

                prev_s = s_next
                s = s_next

            elif phase in ("auction","A"):
                # no trading in the auction (theoretical benchmark)
                noop = _build_auction_noop(env)
                if hasattr(env, "step_auction"):
                    s, r, done, info = _normalize_step_out(env.step_auction(noop))
                else:
                    s, r, done, info = _normalize_step_out(env.step(noop))
                # ignore auction rewards to stay pure (don’t add r)

            else:
                # unknown phase: safe advance
                s, r, done, info = _normalize_step_out(env.step((0.0,0.0,0.0)))

        bm_returns.append(float(ret))

    return bm_returns
