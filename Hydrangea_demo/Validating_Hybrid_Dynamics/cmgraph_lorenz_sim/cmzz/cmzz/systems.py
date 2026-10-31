"""Dynamical systems with a slowly drifting parameter.

Each system supplies a vectorised vector field f(X, lam) where X has shape (..., d)
and lam broadcasts against X[..., 0].  The time-tau map phi_tau(x; lam) is computed
with fixed-step RK4 in numpy; one discrete step of the model is one application of it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DynamicalSystem:
    """Base class.  Subclasses implement `vector_field` and set the metadata."""

    name: str = "system"
    dim: int = 2
    bounds: np.ndarray = None          # (d, 2) domain box
    param_name: str = "lam"
    param_range: tuple = (0.0, 1.0)    # parameter value at t = 0 and t = T
    max_substep: float = 0.0125        # RK4 step size cap
    meta: dict = field(default_factory=dict)

    # --- dynamics ---------------------------------------------------------
    def vector_field(self, X: np.ndarray, lam) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def param_at(self, t, T: float = 1.0):
        """Linear parameter schedule lam(t) over [0, T]."""
        a, b = self.param_range
        return a + (b - a) * (np.asarray(t, dtype=float) / T)

    def flow(self, X: np.ndarray, lam, tau: float) -> np.ndarray:
        """phi_tau(X; lam) by RK4.  X (..., d); lam scalar or broadcastable to X[..., 0]."""
        n = max(1, int(math.ceil(tau / self.max_substep)))
        h = tau / n
        lam = np.asarray(lam, dtype=float)
        f = self.vector_field
        for _ in range(n):
            k1 = f(X, lam)
            k2 = f(X + 0.5 * h * k1, lam)
            k3 = f(X + 0.5 * h * k2, lam)
            k4 = f(X + h * k3, lam)
            X = X + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return X

    def trajectory(self, x0, lam, t_total: float, dt: float = 0.01) -> np.ndarray:
        """A sampled orbit (for drawing only)."""
        steps = int(round(t_total / dt))
        out = [np.asarray(x0, float)]
        x = out[0]
        for _ in range(steps):
            x = self.flow(x, lam, dt)
            out.append(x)
        return np.array(out)


class Hopf2D(DynamicalSystem):
    """Hopf normal form, the motivating example of Dey-Mrozek-Slechta:
    x' = -y + x(lam - x^2 - y^2),  y' = x + y(lam - x^2 - y^2).
    Supercritical Hopf bifurcation at lam = 0 (stable focus -> repelling focus +
    attracting periodic orbit of radius sqrt(lam))."""

    def __init__(self, bounds=((-1.5, 1.5), (-1.5, 1.5)), param_range=(-1.0, 1.0),
                 max_substep=0.0125):
        super().__init__(name="Hopf2D", dim=2, bounds=np.array(bounds, float),
                         param_name="λ", param_range=param_range, max_substep=max_substep,
                         meta={"bifurcation_param": 0.0,
                               "bifurcation": "supercritical Hopf at λ = 0"})

    def vector_field(self, X, lam):
        x, y = X[..., 0], X[..., 1]
        s = lam - (x * x + y * y)
        return np.stack([-y + x * s, x + y * s], axis=-1)


class Lorenz3D(DynamicalSystem):
    """Lorenz system with drifting rho (sigma = 10, beta = 8/3).
    rho = 1: pitchfork (origin -> saddle, two attracting equilibria C+/- born);
    rho ~ 13.93: homoclinic explosion; rho ~ 24.74: subcritical Hopf of C+/-."""

    def __init__(self, bounds=((-6.0, 6.0), (-6.0, 6.0), (-1.0, 9.0)), param_range=(0.2, 6.0),
                 sigma=10.0, beta=8.0 / 3.0, max_substep=0.005):
        super().__init__(name="Lorenz3D", dim=3, bounds=np.array(bounds, float),
                         param_name="ρ", param_range=param_range, max_substep=max_substep,
                         meta={"bifurcation_param": 1.0,
                               "bifurcation": "pitchfork at ρ = 1"})
        self.sigma, self.beta = sigma, beta

    def vector_field(self, X, rho):
        x, y, z = X[..., 0], X[..., 1], X[..., 2]
        return np.stack([self.sigma * (y - x), x * (rho - z) - y, x * y - self.beta * z], axis=-1)


class ODESystem(DynamicalSystem):
    """Generic n-dimensional system from a user vector field f(X, lam)."""

    def __init__(self, f, bounds, param_range=(0.0, 1.0), name="ODE", param_name="λ",
                 max_substep=0.01, **meta):
        bounds = np.array(bounds, float)
        super().__init__(name=name, dim=bounds.shape[0], bounds=bounds, param_name=param_name,
                         param_range=param_range, max_substep=max_substep, meta=meta)
        self._f = f

    def vector_field(self, X, lam):
        return self._f(X, lam)
