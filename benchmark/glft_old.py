from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np

class GLFTBenchmark:
    r"""
    Guéant–Lehalle–Fernandez-Tapia (GLFT) benchmark for optimal liquidation on the
    CLOB phase (risk-neutral limit γ→0, liquidate by T=τ_op-1; remaining shares at S_{T}^{mid}).

    Model recap (your summary):
      1) dS_t^{mid} = σ dB_t
      2) q_t = -N_t^a (unit transaction size), inventory bound Q = I_0
      3) λ^a(δ) = A e^{-k δ},     δ_t = S_t^• - S_t^{mid}
      4) dX_t = (S_t^{mid} + δ_t) dN_t^a
      5) Horizon T = τ_op - 1
      6) Maximize E[-exp(-γ(X_T + q_T S_T^{mid}))] over admissible quotes
      7–9) Let (v_q) solve v_q'(t) = α q^2 v_q(t) - η v_{q-1}(t),  v_q(T)=1,  q=0..Q
           with α = (k γ σ^2)/2,  η = A (1+γ/k)^{-(1 + k/γ)}.
           Writing v(t) = [v_0(t),...,v_Q(t)]ᵗ, we get v(t) = exp(-M (T-t)) e,
           where M is lower–bidiagonal with diag M_{q,q} = α q^2 and
           subdiag M_{q,q-1} = -η, and e = (1,...,1)ᵗ.
     10) Optimal quotes:
           δ^{*}(t,q) = (1/k) ln( v_q(t) / v_{q-1}(t) ) + (1/γ) ln(1 + γ/k).
         Risk-neutral limit (γ→0): δ^{*}(t,q) = (1/k) ln( v_q(t) / v_{q-1}(t) ) + 1/k.

    This class provides:
      - v_q(t) for discrete integer t ∈ {0,...,T}
      - δ^{*}(t,q) (handles q=0 by returning +∞, i.e., no sell incentive)
      - policy() mapping (t,q) → δ^{*}(t,q)
      - expected_value(x0,q0,s0): risk-neutral value proxy
            V*_0 = x0 + q0 s0 + (1/k) ln v_{q0}(0)
        (since u(0,x,q,s) = -exp(-γ [ x + q s + (1/k) ln v_q(0) ]) and γ→0 ⇒ CE → V*_0)

    Notes
    -----
    • We use γ→0 by default:
        α → 0,   η → A / e.
      You can pass a small positive γ to explore risk aversion; we keep the general formulas.
    • We solve v(t) with an eigen-decomposition of M (no SciPy dependency).
    • Time is treated on an integer grid; v(t) is evaluated exactly via expm(-M (T-t)).
    """

    def __init__(
        self,
        A: float,
        k: float,
        sigma: float,
        Q: int,
        T: int,
        alpha_tick: float,           # kept for completeness; not used inside formulas
        gamma: float = 0.0,          # default to risk-neutral limit
        eps: float = 1e-12,          # numerical guard
    ):
        self.A = float(A)
        self.k = float(k)
        self.sigma = float(sigma)
        self.Q = int(Q)
        self.T = int(T)
        self.alpha_tick = float(alpha_tick)
        self.gamma = float(max(0.0, gamma))
        self.eps = float(eps)

        # Coefficients α, η (handle γ→0 with the proper limits)
        if self.gamma == 0.0:
            self.alpha_coef = 0.0
            self.eta = self.A / np.e
        else:
            self.alpha_coef = (self.k * self.gamma * (self.sigma ** 2)) / 2.0
            self.eta = self.A * (1.0 + self.gamma / self.k) ** (-(1.0 + self.k / self.gamma))

        # Build the (Q+1)x(Q+1) lower–bidiagonal matrix M
        self.M = self._build_M(self.Q, self.alpha_coef, self.eta)

        # Precompute eigen-decomposition for fast expm(-M τ)·e
        self._eigvals, self._eigvecs = np.linalg.eig(self.M)
        self._eigvecs_inv = np.linalg.inv(self._eigvecs)

        # Constant vector e = ones(Q+1)
        self._e = np.ones(self.Q + 1, dtype=float)

        # Cache for v(t): dict t -> np.ndarray length (Q+1)
        self._v_cache: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def v_q(self, t: int) -> np.ndarray:
        """
        Return the vector [v_0(t),...,v_Q(t)]ᵗ using v(t) = exp(-M (T - t)) e.

        Args:
            t: integer in [0, T]

        Returns:
            np.ndarray of shape (Q+1,)
        """
        t = int(t)
        if not (0 <= t <= self.T):
            raise ValueError(f"t must be in [0, {self.T}]")

        if t in self._v_cache:
            return self._v_cache[t]

        tau = float(self.T - t)  # time-to-go
        v = self._apply_expm_times_vec(self.M, tau, self._e)
        # Guard against tiny negative roundoff
        v = np.maximum(v, self.eps)
        self._v_cache[t] = v
        return v

    def delta_star(self, t: int, q: int, gamma: Optional[float] = None) -> float:
        """
        δ^{*}(t,q) = (1/k) ln( v_q(t) / v_{q-1}(t) ) + (1/γ) ln(1 + γ/k)
        Risk-neutral limit γ→0: δ^{*} = (1/k) ln(v_q/v_{q-1}) + 1/k.

        For q=0 (no shares sold yet), we return +∞ (no sell incentive needed).
        """
        if q <= 0:
            return float("inf")
        if q > self.Q:
            q = self.Q

        v = self.v_q(t)
        num = float(v[q])
        den = float(v[q - 1])
        num = max(num, self.eps)
        den = max(den, self.eps)

        if gamma is None:
            gamma = self.gamma

        core = (1.0 / self.k) * np.log(num / den)

        if gamma == 0.0:
            offset = 1.0 / self.k
        else:
            offset = (1.0 / gamma) * np.log(1.0 + gamma / self.k)

        return float(core + offset)

    def policy(self) -> Dict[Tuple[int, int], float]:
        """
        Return a dict mapping (t,q) -> δ^{*}(t,q) for t=0..T, q=0..Q.
        """
        pi: Dict[Tuple[int, int], float] = {}
        for t in range(self.T + 1):
            v = self.v_q(t)  # cache reuse
            for q in range(self.Q + 1):
                pi[(t, q)] = self.delta_star(t, q)
        return pi

    def expected_value(self, x0: float, q0: int, s0: float) -> float:
        """
        GLFT value proxy at t=0 for regret:
            V*_0 = x0 + q0 s0 + (1/k) ln v_{q0}(0)
        (This is the certainty-equivalent as γ→0, since u = -exp(-γ V*) ).
        """
        q0 = max(0, min(int(q0), self.Q))
        v0 = self.v_q(0)[q0]
        v0 = max(float(v0), self.eps)
        return float(x0 + q0 * s0 + (1.0 / self.k) * np.log(v0))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_M(Q: int, alpha_coef: float, eta: float) -> np.ndarray:
        """
        Build (Q+1)x(Q+1) matrix M with:
            M_{q,q}     = α q^2
            M_{q,q-1}   = -η   for q>=1
            others      = 0
        The index q corresponds to inventory level (0..Q).
        """
        n = Q + 1
        M = np.zeros((n, n), dtype=float)
        for q in range(n):
            M[q, q] = alpha_coef * (q ** 2)
            if q >= 1:
                M[q, q - 1] = -eta
        return M

    def _apply_expm_times_vec(self, M: np.ndarray, tau: float, vec: np.ndarray) -> np.ndarray:
        """
        Compute exp(-M τ) @ vec using the cached eigen-decomposition of M.
        For small/medium (Q+1), eigen-based expm is accurate and SciPy-free.
        """
        # exp(-Λ τ)
        exp_diag = np.exp(-self._eigvals * tau)
        # P exp(-Λ τ) P^{-1} vec
        tmp = self._eigvecs_inv @ vec
        tmp *= exp_diag
        out = self._eigvecs @ tmp
        # ensure real due to numerics
        return np.real_if_close(out, tol=1e-10)
