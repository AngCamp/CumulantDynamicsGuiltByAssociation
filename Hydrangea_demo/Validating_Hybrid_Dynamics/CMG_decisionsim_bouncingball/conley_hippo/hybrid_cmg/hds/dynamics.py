"""
hds.dynamics -- composable latent vector fields + hybrid (guard/regime) layer.

Latent state x lives in R^3: the first two coordinates carry the attractor
geometry, the third is an auxiliary decaying coordinate (adds a realistic
extra dimension the readout mixes in).

Every field component is a callable  f(X) -> dX  with X of shape (n, 3),
vectorized over trials. Components compose:

  Sum([f1, f2, ...])                  additive superposition
  Mixture([f1, f2], centers, width)   soft partition of space: each component
                                      rules near its center (smooth basins;
                                      boundaries between basins become
                                      separatrices / saddle regions)
  SpeedMap(f, region, factor)         multiplies speed inside a region
                                      (factor < 1 -> low-energy DWELL zone
                                      where no attractor is particularly
                                      active)
  Gated(f, region)                    f only acts inside a soft region

Primitive components:
  NestedRings      the Sim A field: outer ring + thin counter-rotating band
  RingAttractor    limit cycle with radius, angular speed, radial stiffness,
                   optional speed modulation along the ring
  PointAttractor   stable node
  Saddle           one unstable + one stable direction
  Channel          drift along a segment inside a tube (flow BETWEEN
                   attractors -- heteroclinic-like routes)
  AuxDecay         OU-style decay of the auxiliary 3rd coordinate

Hybrid layer:
  Mode(name, field_gain, rate_gain, mean_duration, exit_to)
  Guard(name, condition, hazard, jump, to_mode, from_modes, refractory)
     condition(X, lam) -> bool  (lam = expected neural rates, so a guard can
     be "these neurons are sufficiently active")
     jump(X, rng) -> X'          instantaneous state map applied at the switch

Distortions of attractor GEOMETRY (not dynamics) live on the readout side,
see hds.neurons (warps applied to x before tuning curves).
"""

import numpy as np

# ------------------------------- primitives ----------------------------------


class AuxDecay:
    """dz = -k z on the auxiliary coordinate."""

    def __init__(self, k=3.0):
        self.k = k

    def __call__(self, X):
        d = np.zeros_like(X)
        d[:, 2] = -self.k * X[:, 2]
        return d


class NestedRings:
    """Exact Sim A field. rho' = -A (rho-r_in)(rho-r_bar)(rho-r_out);
    theta' = w_out outside the barrier circle, w_in inside (reversed)."""

    def __init__(self, r_in=0.72, r_bar=0.86, r_out=1.0, w_out=3.0,
                 w_in=-12.0, A=70.0):
        self.__dict__.update(locals())
        del self.__dict__['self']

    def __call__(self, X):
        x, y = X[:, 0], X[:, 1]
        rho = np.sqrt(x * x + y * y) + 1e-12
        drho = -self.A * (rho - self.r_in) * (rho - self.r_bar) * \
            (rho - self.r_out)
        w = np.where(rho > self.r_bar, self.w_out, self.w_in)
        d = np.zeros_like(X)
        d[:, 0] = drho * x / rho - w * y
        d[:, 1] = drho * y / rho + w * x
        return d

    def ring_of(self, X):
        rho = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
        return np.where(rho > self.r_bar, 1, 0)        # 1 outer, 0 inner


class RingAttractor:
    """Limit cycle centered at c (in the x-y plane).
    speed_mod(theta) multiplies the angular speed (distorts the flow ALONG the
    ring without changing its topology; e.g. slow sector = dwell on the ring)."""

    def __init__(self, center=(0, 0), radius=1.0, omega=3.0, stiffness=8.0,
                 speed_mod=None):
        self.c = np.asarray(center, float)
        self.r, self.w, self.k = radius, omega, stiffness
        self.speed_mod = speed_mod

    def __call__(self, X):
        x, y = X[:, 0] - self.c[0], X[:, 1] - self.c[1]
        rho = np.sqrt(x * x + y * y) + 1e-12
        th = np.arctan2(y, x)
        w = self.w * (self.speed_mod(th) if self.speed_mod else 1.0)
        drho = -self.k * (rho - self.r)
        d = np.zeros_like(X)
        d[:, 0] = drho * x / rho - w * y
        d[:, 1] = drho * y / rho + w * x
        return d


class PointAttractor:
    def __init__(self, center=(0, 0), k=4.0):
        self.c = np.asarray(center, float)
        self.k = k

    def __call__(self, X):
        d = np.zeros_like(X)
        d[:, :2] = -self.k * (X[:, :2] - self.c)
        return d


class Saddle:
    """Linear saddle at c: grows along u (rate lam_u), contracts along the
    orthogonal direction (rate lam_s)."""

    def __init__(self, center=(0, 0), unstable_dir=(1, 0), lam_u=2.0,
                 lam_s=4.0):
        self.c = np.asarray(center, float)
        u = np.asarray(unstable_dir, float)
        self.u = u / np.linalg.norm(u)
        self.s = np.array([-self.u[1], self.u[0]])
        self.lu, self.ls = lam_u, lam_s

    def __call__(self, X):
        r = X[:, :2] - self.c
        a = r @ self.u
        b = r @ self.s
        d = np.zeros_like(X)
        d[:, :2] = np.outer(self.lu * a, self.u) - np.outer(self.ls * b, self.s)
        return d


class Channel:
    """Drift from point a to point b inside a tube of given width: models a
    directed flow route between attractors. Strength falls off as a Gaussian
    of the distance to the segment."""

    def __init__(self, a, b, speed=1.0, width=0.15, pull=3.0):
        self.a, self.b = np.asarray(a, float), np.asarray(b, float)
        seg = self.b - self.a
        self.L = np.linalg.norm(seg)
        self.t = seg / self.L
        self.n = np.array([-self.t[1], self.t[0]])
        self.v, self.w, self.pull = speed, width, pull

    def __call__(self, X):
        r = X[:, :2] - self.a
        s = np.clip(r @ self.t, 0, self.L)
        off = r - np.outer(s, self.t)
        dist = np.linalg.norm(off, axis=1)
        g = np.exp(-0.5 * (dist / self.w) ** 2) * (r @ self.t > -self.w) * \
            (r @ self.t < self.L + self.w)
        d = np.zeros_like(X)
        d[:, :2] = g[:, None] * (self.v * self.t - self.pull * off)
        return d


# ------------------------------- combinators ---------------------------------


class Sum:
    def __init__(self, comps):
        self.comps = comps

    def __call__(self, X):
        return sum(c(X) for c in self.comps)


class Mixture:
    """Soft partition: weights w_i = softmax(-|x-c_i|^2 / (2 width^2))."""

    def __init__(self, comps, centers, width=0.5):
        self.comps = comps
        self.C = np.asarray(centers, float)
        self.w = width

    def weights(self, X):
        d2 = ((X[:, None, :2] - self.C[None]) ** 2).sum(-1)
        e = -d2 / (2 * self.w ** 2)
        e -= e.max(1, keepdims=True)
        W = np.exp(e)
        return W / W.sum(1, keepdims=True)

    def __call__(self, X):
        W = self.weights(X)
        return sum(W[:, i:i + 1] * c(X) for i, c in enumerate(self.comps))


class SpeedMap:
    """Multiply the field by factor inside region(X) (soft, in [0,1]).
    factor < 1 -> low-energy dwell state."""

    def __init__(self, field, region, factor=0.15):
        self.f, self.region, self.factor = field, region, factor

    def __call__(self, X):
        g = self.region(X)
        return self.f(X) * (1 - g * (1 - self.factor))[:, None]


class Gated:
    def __init__(self, field, region):
        self.f, self.region = field, region

    def __call__(self, X):
        return self.f(X) * self.region(X)[:, None]


def disk_region(center, radius, soft=0.05):
    c = np.asarray(center, float)

    def reg(X):
        d = np.linalg.norm(X[:, :2] - c, axis=1)
        return 1.0 / (1.0 + np.exp((d - radius) / soft))
    return reg


# ------------------------------- hybrid layer ---------------------------------


class Mode:
    def __init__(self, name, field_gain=1.0, rate_gain=1.0,
                 mean_duration=None, exit_to=None, exit_jump=None):
        self.name = name
        self.field_gain = field_gain
        self.rate_gain = rate_gain
        self.mean_duration = mean_duration     # seconds; None = persistent
        self.exit_to = exit_to                 # mode name on expiry
        self.exit_jump = exit_jump             # optional state map on expiry


class Guard:
    def __init__(self, name, condition, hazard, jump, to_mode,
                 from_modes=("base",), refractory=0.3):
        self.name = name
        self.condition = condition             # (X, lam) -> bool (n,);
        #   lam(idx) returns expected rates (n, len(idx)) of neurons idx
        self.hazard = hazard                   # per second while condition
        self.jump = jump                       # (X, rng) -> X'
        self.to_mode = to_mode
        self.from_modes = from_modes
        self.refractory = refractory


def phase_advance(dtheta):
    """Jump map: move the state ALONG its ring by dtheta (rad), keeping its
    radius -- 'takes the system to a place on a trajectory'."""
    def jump(X, rng):
        Y = X.copy()
        c, s = np.cos(dtheta), np.sin(dtheta)
        Y[:, 0] = c * X[:, 0] - s * X[:, 1]
        Y[:, 1] = s * X[:, 0] + c * X[:, 1]
        return Y
    return jump


def kick(vec):
    """Jump map: constant displacement."""
    v = np.asarray(vec, float)

    def jump(X, rng):
        return X + v
    return jump


# ------------------------------- simulator ------------------------------------


class HybridSystem:
    """field: base vector field. modes: dict name->Mode (must include 'base').
    guards: list of Guard. readout: hds.neurons.Readout (used for guard
    conditions on neural activity and for rate gain)."""

    def __init__(self, field, modes, guards, sigma=0.012, sigma_aux=0.05,
                 shared_tau=0.005, shared_sd=0.15):
        self.field = field
        self.modes = modes
        self.mode_names = list(modes)
        self.guards = guards
        self.sigma, self.sigma_aux = sigma, sigma_aux
        self.shared_tau, self.shared_sd = shared_tau, shared_sd

    def gained_field(self, X, mode_idx):
        g = np.array([self.modes[m].field_gain for m in self.mode_names])
        return self.field(X) * g[mode_idx][:, None]

    def simulate(self, x0, T, readout=None, dt=0.001, base=0.005, burn=3.0,
                 seed=0):
        """x0: (n,3) initial states. Returns dict with base-resolution
        recordings (n, K, ...) and a list of events
        (trial, base_bin, kind, guard/mode name)."""
        rng = np.random.default_rng(seed)
        n = x0.shape[0]
        X = x0.astype(float).copy()
        steps, burn_s, sub = int(T / dt), int(burn / dt), int(round(base / dt))
        K = (steps - burn_s) // sub
        mi = np.zeros(n, int)                       # mode index (0 = base)
        base_idx = self.mode_names.index('base')
        mi[:] = base_idx
        timer = np.full(n, np.inf)                  # time left in mode
        refr = np.zeros(n)                          # guard refractory
        eta = np.zeros(n)                           # shared gain noise
        rec = dict(X=np.zeros((n, K, 3)), eta=np.zeros((n, K)),
                   mode=np.zeros((n, K), np.int8),
                   guard=np.zeros((n, K), np.int8))
        acc = np.zeros((n, 3))
        acc_e = np.zeros(n)
        g_any = np.zeros(n, bool)
        events = []
        sq = np.sqrt(dt)
        k = 0
        for s in range(steps):
            # --- flow: RK4 drift + Euler-Maruyama noise ---
            # (explicit Euler inflates the radius of fast rotations by
            #  ~(omega dt)^2 / 2 per step -- at 36 rad/s that outruns the weak
            #  cubic radial force and pushes the state across the barrier)
            k1 = self.gained_field(X, mi)
            k2 = self.gained_field(X + 0.5 * dt * k1, mi)
            k3 = self.gained_field(X + 0.5 * dt * k2, mi)
            k4 = self.gained_field(X + dt * k3, mi)
            X = X + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            X[:, :2] += self.sigma * sq * rng.normal(size=(n, 2))
            X[:, 2] += self.sigma_aux * sq * rng.normal(size=n)
            eta += -eta / self.shared_tau * dt + self.shared_sd * \
                np.sqrt(2 / self.shared_tau) * sq * rng.normal(size=n)
            refr = np.maximum(refr - dt, 0)
            # --- mode expiry ---
            timer -= dt
            exp_ = np.isfinite(timer) & (timer <= 0)
            for i in np.flatnonzero(exp_):
                m = self.modes[self.mode_names[mi[i]]]
                if m.exit_jump is not None:
                    X[i:i + 1] = m.exit_jump(X[i:i + 1], rng)
                mi[i] = self.mode_names.index(m.exit_to or 'base')
                timer[i] = np.inf
                refr[i] = max(refr[i], 0.3)
                if s >= burn_s:
                    events.append((i, (s - burn_s) // sub, 'return',
                                   m.name))
            # --- guards ---
            def lam(idx, X=X, eta=eta, mi=mi):
                return readout.expected_rates(X, eta, mi, self, idx)
            g_flag = np.zeros(n, bool)
            for gd in self.guards:
                ok = np.isin(np.array(self.mode_names)[mi], gd.from_modes)
                cond = gd.condition(X, lam) & ok
                g_flag |= cond
                fire = cond & (refr <= 0) & \
                    (rng.random(n) < gd.hazard * dt)
                for i in np.flatnonzero(fire):
                    X[i:i + 1] = gd.jump(X[i:i + 1], rng)
                    mi[i] = self.mode_names.index(gd.to_mode)
                    md = self.modes[gd.to_mode]
                    timer[i] = rng.exponential(md.mean_duration) \
                        if md.mean_duration else np.inf
                    refr[i] = gd.refractory
                    if s >= burn_s:
                        events.append((i, (s - burn_s) // sub, 'jump',
                                       gd.name))
            g_any |= g_flag
            # --- record ---
            if s >= burn_s:
                acc += X
                acc_e += eta
                if (s - burn_s + 1) % sub == 0 and k < K:
                    rec['X'][:, k] = acc / sub
                    rec['eta'][:, k] = acc_e / sub
                    rec['mode'][:, k] = mi
                    rec['guard'][:, k] = g_any
                    acc[:] = 0
                    acc_e[:] = 0
                    g_any[:] = False
                    k += 1
        rec['events'] = events
        rec['base'] = base
        rec['mode_names'] = self.mode_names
        return rec

    def lipschitz_true(self, X, mode_idx, eps=1e-4):
        """Spectral norm of the Jacobian of the GAINED field at each X
        (ground-truth local Lipschitz constant)."""
        n = X.shape[0]
        J = np.zeros((n, 3, 3))
        for j in range(3):
            E = np.zeros(3)
            E[j] = eps
            J[:, :, j] = (self.gained_field(X + E, mode_idx) -
                          self.gained_field(X - E, mode_idx)) / (2 * eps)
        return np.linalg.norm(J, ord=2, axis=(1, 2))
