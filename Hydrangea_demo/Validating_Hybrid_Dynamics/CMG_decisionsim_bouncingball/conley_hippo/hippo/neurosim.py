"""
neurosim.py
-----------
Generic building blocks for stochastic (escape-noise) leaky integrate-and-fire
populations with binomial spike generation.

Nothing in this file refers to the hippocampus. The scenario file imports these
classes and composes them.

Model summary
-------------
Each neuron keeps a membrane variable v that integrates its input current with a
leak (leaky integrate-and-fire). Instead of a hard threshold, the instantaneous
firing rate is an escape function of the distance from threshold,

    r(t) = r0 * exp(beta * (v(t) - v_thresh))             [Hz]

The per-bin firing probability is p = r * dt, multiplied by a time-varying gain
(oscillations, vetoes), plus an additive burst probability trace, then clipped
to [0, p_max]. With dt = 1 ms, p_max = 1 corresponds to the 1 kHz ceiling. A
spike in a bin is a Bernoulli (binomial, n = 1) draw with probability p. On a
spike v is reset and the neuron enters an absolute refractory period.

Bursts are modelled as a binomial event conditional on a spike: with probability
burst_prob the neuron schedules (burst_n - 1) additional spikes at burst_isi
intervals. Those scheduled spikes enter the probability trace as an additive
term, so the burst trace is a separate component that is summed into the final
probability.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

import numpy as np

MAX_RATE_HZ = 1000.0


# --------------------------------------------------------------------------- #
# 1. Binomial spike generation
# --------------------------------------------------------------------------- #
class BinomialSpiker:
    """Per-bin Bernoulli (binomial n=1) spike draws from a probability vector."""

    def __init__(self, rng: np.random.Generator, dt_ms: float):
        self.rng = rng
        self.dt_s = dt_ms / 1000.0
        self.p_max = float(min(1.0, MAX_RATE_HZ * self.dt_s))

    def rate_to_prob(self, rate_hz):
        return np.clip(np.asarray(rate_hz, dtype=float) * self.dt_s, 0.0, self.p_max)

    def clip(self, p):
        return np.clip(p, 0.0, self.p_max)

    def sample(self, p) -> np.ndarray:
        p = self.clip(p)
        return self.rng.random(np.shape(p)) < p


# --------------------------------------------------------------------------- #
# 2. Burst trace (additive probability component)
# --------------------------------------------------------------------------- #
class BurstTrace:
    """Schedules extra spikes after a burst-triggering spike.

    trigger(): binomial draw per spiking neuron; successes schedule a burst.
    step():    returns the additive probability trace for the current bin
               (1.0 where a scheduled burst spike falls, else 0.0).
    """

    def __init__(self, n: int, dt_ms: float, rng: np.random.Generator):
        self.n = n
        self.dt_ms = dt_ms
        self.rng = rng
        self.pending = np.zeros(n, dtype=int)     # spikes still to emit
        self.timer = np.zeros(n, dtype=float)     # ms until next burst spike
        self.last_emitted = np.zeros(n, dtype=bool)

    def trigger(self, spiked: np.ndarray, burst_prob, burst_n, burst_isi_ms):
        burst_prob = np.broadcast_to(np.asarray(burst_prob, float), (self.n,))
        burst_n = np.broadcast_to(np.asarray(burst_n, int), (self.n,))
        burst_isi_ms = np.broadcast_to(np.asarray(burst_isi_ms, float), (self.n,))
        # only spikes that were not themselves scheduled by a burst can start one
        candidates = spiked & (self.pending == 0) & ~self.last_emitted
        draw = self.rng.random(self.n) < burst_prob
        start = candidates & draw & (burst_n > 1)
        self.pending[start] = burst_n[start] - 1
        self.timer[start] = burst_isi_ms[start]

    def step(self, burst_isi_ms) -> np.ndarray:
        burst_isi_ms = np.broadcast_to(np.asarray(burst_isi_ms, float), (self.n,))
        add = np.zeros(self.n, dtype=float)
        active = self.pending > 0
        self.timer[active] -= self.dt_ms
        fire = active & (self.timer <= 0.0)
        add[fire] = 1.0
        self.last_emitted = fire
        self.pending[fire] -= 1
        self.timer[fire] = burst_isi_ms[fire]
        return add


# --------------------------------------------------------------------------- #
# 3. Probability trace: leaky integration -> escape rate -> probability
# --------------------------------------------------------------------------- #
class ProbabilityTrace:
    """Turns input current, multiplicative gain and additive burst probability
    into a per-bin firing probability for a population."""

    def __init__(self, n: int, dt_ms: float, spiker: BinomialSpiker,
                 v_thresh: float = 1.0, v_reset: float = 0.0,
                 beta: float = 4.0, r0_hz: float = 20.0,
                 refractory_ms: float = 2.0):
        self.n = n
        self.dt_ms = dt_ms
        self.spiker = spiker
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.beta = beta
        self.r0 = r0_hz
        self.refractory_ms = refractory_ms
        self.v = np.full(n, v_reset, dtype=float)
        self.refr = np.zeros(n, dtype=float)
        # last components, kept for inspection / plotting
        self.last = {}

    def step(self, current: np.ndarray, tau_m_ms, mult_gain, add_prob) -> np.ndarray:
        tau = np.broadcast_to(np.asarray(tau_m_ms, float), (self.n,))
        self.v += (self.dt_ms / tau) * (-self.v + current)
        rate = self.r0 * np.exp(self.beta * (self.v - self.v_thresh))
        rate = np.minimum(rate, MAX_RATE_HZ)
        p_base = self.spiker.rate_to_prob(rate)
        p = p_base * mult_gain + add_prob
        p = np.where(self.refr > 0.0, 0.0, p)
        p = self.spiker.clip(p)
        self.last = dict(v=self.v.copy(), rate=rate, p_base=p_base,
                         gain=np.broadcast_to(mult_gain, (self.n,)).copy(),
                         burst=np.asarray(add_prob).copy(), p=p)
        return p

    def register_spikes(self, spiked: np.ndarray):
        self.v[spiked] = self.v_reset
        self.refr[spiked] = self.refractory_ms
        self.refr -= self.dt_ms
        np.maximum(self.refr, 0.0, out=self.refr)


# --------------------------------------------------------------------------- #
# 4. Stimuli and tuning
# --------------------------------------------------------------------------- #
class Stimulus:
    """A real-valued variable of given dimensionality with an evidence weight in [0, 1]."""

    def __init__(self, name: str, dim: int, evidence: float = 1.0):
        self.name = name
        self.dim = dim
        self.evidence = float(np.clip(evidence, 0.0, 1.0))
        self.value = np.zeros(dim, dtype=float)

    def set(self, value, evidence: Optional[float] = None):
        self.value = np.asarray(value, dtype=float).reshape(self.dim)
        if evidence is not None:
            self.evidence = float(np.clip(evidence, 0.0, 1.0))


class DriveStimulus(Stimulus):
    """A stimulus whose dimensionality equals the number of target neurons and
    whose value is a direct per-neuron drive. Used for externally scheduled
    patterns (for instance replayed sequences) without the tuning layer needing
    to know where the pattern came from."""

    def __init__(self, name: str, n: int, evidence: float = 1.0):
        super().__init__(name, n, evidence)

    def drive(self) -> np.ndarray:
        return self.value * self.evidence


class GaussianTuning:
    """Gaussian tuning of N neurons to a stimulus of dimension `dim`.

    centers : (N, dim)   preferred stimulus value
    widths  : (N,)       standard deviation in stimulus units
    amps    : (N,)       peak drive (current units); 0 disables tuning
    burst_coupling : (N,) multiplies the burst probability by (1 + c * normalised drive)
    """

    def __init__(self, centers, widths, amps, burst_coupling=0.0):
        self.centers = np.asarray(centers, float)
        self.n, self.dim = self.centers.shape
        self.widths = np.broadcast_to(np.asarray(widths, float), (self.n,)).copy()
        self.amps = np.broadcast_to(np.asarray(amps, float), (self.n,)).copy()
        self.burst_coupling = np.broadcast_to(np.asarray(burst_coupling, float), (self.n,)).copy()
        self.last_norm = np.zeros(self.n)

    def response(self, stimulus: Stimulus) -> np.ndarray:
        """Normalised tuning response in [0, 1] (before evidence and amplitude)."""
        d2 = np.sum((self.centers - stimulus.value[None, :]) ** 2, axis=1)
        self.last_norm = np.exp(-0.5 * d2 / (self.widths ** 2))
        return self.last_norm

    def drive(self, stimulus: Stimulus) -> np.ndarray:
        return self.amps * self.response(stimulus) * stimulus.evidence

    def burst_factor(self) -> np.ndarray:
        return 1.0 + self.burst_coupling * self.last_norm


# --------------------------------------------------------------------------- #
# 5. State-dependent background input
# --------------------------------------------------------------------------- #
@dataclass
class StateParams:
    """Per-population parameters that a network state imposes."""
    baseline: float = 0.0          # constant current
    noise_sd: float = 0.0          # white current noise (per bin)
    stim_gain: float = 1.0         # multiplies tuned stimulus drive
    osc_depth: float = 0.0         # depth of oscillatory multiplicative gain
    osc_phase: float = 0.0         # preferred phase (rad) of the oscillation
    mult_gain: float = 1.0         # extra constant multiplicative gain
    burst_prob: float = 0.0        # P(burst | spike)
    burst_n: int = 1               # spikes per burst
    burst_isi_ms: float = 6.0      # intra-burst interval
    tau_m_ms: Optional[float] = None  # membrane time constant override
    noise_tau_ms: float = 0.0      # 0 = white per-bin noise; >0 = Ornstein-Uhlenbeck with this correlation time

    @staticmethod
    def from_dict(d: dict) -> "StateParams":
        return StateParams(**{k: v for k, v in d.items() if k in StateParams.__dataclass_fields__})


class BackgroundInput:
    """Maps a state name to StateParams for one population.

    The background input is the population-level mechanism through which a
    network state (for instance theta or ripple) injects baseline drive,
    oscillatory gain, noise and burst propensity."""

    def __init__(self, params: Dict[str, StateParams], rng: np.random.Generator, n: int,
                 dt_ms: float = 1.0):
        self.params = params
        self.rng = rng
        self.n = n
        self.dt_ms = dt_ms
        self.ou = np.zeros(n)

    def get(self, state: str) -> StateParams:
        return self.params[state]

    def noise(self, state: str) -> np.ndarray:
        """Current noise. With noise_tau_ms > 0 an OU process whose standing
        standard deviation equals noise_sd; otherwise white noise per bin."""
        sp = self.params[state]
        sd = sp.noise_sd
        if sd <= 0:
            self.ou *= 0.0
            return np.zeros(self.n)
        if sp.noise_tau_ms > 0:
            a = np.exp(-self.dt_ms / sp.noise_tau_ms)
            self.ou = a * self.ou + sd * np.sqrt(1.0 - a * a) * self.rng.normal(0.0, 1.0, self.n)
            return self.ou
        return self.rng.normal(0.0, sd, self.n)


class Oscillator:
    """Phase accumulator; frequency can be changed every step."""

    def __init__(self, freq_hz: float, dt_ms: float, phase: float = 0.0):
        self.freq = freq_hz
        self.dt_s = dt_ms / 1000.0
        self.phase = phase

    def step(self, freq_hz: Optional[float] = None) -> float:
        if freq_hz is not None:
            self.freq = freq_hz
        self.phase = (self.phase + 2.0 * np.pi * self.freq * self.dt_s) % (2.0 * np.pi)
        return self.phase

    def gain(self, depth, pref_phase) -> np.ndarray:
        """Multiplicative gain 1 + depth * cos(phase - pref)."""
        return 1.0 + np.asarray(depth) * np.cos(self.phase - np.asarray(pref_phase))


# --------------------------------------------------------------------------- #
# 6. Projections between populations
# --------------------------------------------------------------------------- #
class Projection:
    """Exponentially filtered synaptic current from pre to post.

    W : (n_pre, n_post) signed weights. s decays with tau and is incremented by
    W^T @ spikes_pre. current() returns the per-post-neuron current."""

    def __init__(self, W: np.ndarray, tau_ms: float, dt_ms: float):
        self.W = np.asarray(W, float)
        self.n_pre, self.n_post = self.W.shape
        self.decay = np.exp(-dt_ms / tau_ms)
        self.s = np.zeros(self.n_post)

    def step(self, pre_spikes: np.ndarray) -> np.ndarray:
        self.s *= self.decay
        if pre_spikes.any():
            self.s += self.W[pre_spikes].sum(axis=0)
        return self.s

    @staticmethod
    def random(n_pre: int, n_post: int, p_conn: float, weight: float,
               rng: np.random.Generator, tau_ms: float, dt_ms: float) -> "Projection":
        mask = rng.random((n_pre, n_post)) < p_conn
        W = mask * weight * rng.lognormal(0.0, 0.3, (n_pre, n_post))
        return Projection(W, tau_ms, dt_ms)

    @staticmethod
    def ring(n: int, sigma_neurons: float, weight: float, tau_ms: float, dt_ms: float,
             exclude_self: bool = True) -> "Projection":
        """Recurrent weights that fall off with circular distance on a ring of n neurons."""
        idx = np.arange(n)
        d = np.abs(idx[:, None] - idx[None, :])
        d = np.minimum(d, n - d)
        W = weight * np.exp(-0.5 * (d / sigma_neurons) ** 2)
        if exclude_self:
            np.fill_diagonal(W, 0.0)
        return Projection(W, tau_ms, dt_ms)


class SpikeTrace:
    """Exponentially filtered population activity (scalar), used for gating terms."""

    def __init__(self, tau_ms: float, dt_ms: float, n: int):
        self.decay = np.exp(-dt_ms / tau_ms)
        self.n = n
        self.x = 0.0

    def step(self, spikes: np.ndarray) -> float:
        self.x = self.x * self.decay + spikes.sum() / self.n
        return self.x


# --------------------------------------------------------------------------- #
# 7. Neuron population
# --------------------------------------------------------------------------- #
class LIFPopulation:
    """A population of escape-noise LIF neurons with binomial spiking and bursts."""

    def __init__(self, name: str, n: int, dt_ms: float, rng: np.random.Generator,
                 tau_m_ms: float = 20.0, v_thresh: float = 1.0, v_reset: float = 0.0,
                 beta: float = 4.0, r0_hz: float = 20.0, refractory_ms: float = 2.0,
                 background: Optional[BackgroundInput] = None,
                 tuning: Optional[GaussianTuning] = None):
        self.name = name
        self.n = n
        self.dt_ms = dt_ms
        self.rng = rng
        self.tau_m = tau_m_ms
        self.spiker = BinomialSpiker(rng, dt_ms)
        self.ptrace = ProbabilityTrace(n, dt_ms, self.spiker, v_thresh, v_reset,
                                       beta, r0_hz, refractory_ms)
        self.bursts = BurstTrace(n, dt_ms, rng)
        self.background = background
        self.tuning = tuning
        self.spikes = np.zeros(n, dtype=bool)

    def step(self, state: str, current: np.ndarray, mult_gain=1.0,
             burst_prob_factor=1.0) -> np.ndarray:
        sp = self.background.get(state)
        tau = self.tau_m if sp.tau_m_ms is None else sp.tau_m_ms
        add = self.bursts.step(sp.burst_isi_ms)
        p = self.ptrace.step(current, tau, np.asarray(mult_gain) * sp.mult_gain, add)
        spiked = self.spiker.sample(p)
        self.ptrace.register_spikes(spiked)
        self.bursts.trigger(spiked, sp.burst_prob * np.asarray(burst_prob_factor),
                            sp.burst_n, sp.burst_isi_ms)
        self.spikes = spiked
        return spiked


# --------------------------------------------------------------------------- #
# 8. States, transitions and events
# --------------------------------------------------------------------------- #
class StateMachine:
    """Discrete states with lognormal dwell times and a transition matrix."""

    def __init__(self, names: Sequence[str], dwell_mean_s: Dict[str, float],
                 transition: Dict[str, Dict[str, float]], rng: np.random.Generator,
                 dt_ms: float, dwell_sigma_log: float = 0.3, initial: Optional[str] = None):
        self.names = list(names)
        self.dwell_mean = dwell_mean_s
        self.sigma = dwell_sigma_log
        self.rng = rng
        self.dt_s = dt_ms / 1000.0
        self.T = np.zeros((len(names), len(names)))
        for i, a in enumerate(self.names):
            for j, b in enumerate(self.names):
                self.T[i, j] = transition.get(a, {}).get(b, 0.0)
        self.T = self.T / np.maximum(self.T.sum(axis=1, keepdims=True), 1e-12)
        self.state = initial if initial is not None else self.names[0]
        self.remaining = self._draw_dwell(self.state)
        self.elapsed = 0.0
        self.dwell_log: List[tuple] = []

    def _draw_dwell(self, state: str) -> float:
        mu = np.log(self.dwell_mean[state]) - 0.5 * self.sigma ** 2
        return float(self.rng.lognormal(mu, self.sigma))

    def step(self) -> tuple:
        self.remaining -= self.dt_s
        self.elapsed += self.dt_s
        changed = False
        if self.remaining <= 0.0:
            i = self.names.index(self.state)
            self.dwell_log.append((self.state, self.elapsed))
            self.state = self.rng.choice(self.names, p=self.T[i])
            self.remaining = self._draw_dwell(self.state)
            self.elapsed = 0.0
            changed = True
        return self.state, changed


class PoissonEvents:
    """Events occurring at a Poisson rate with lognormal duration and gain."""

    def __init__(self, rate_hz: float, dur_median_ms: float, dur_sigma_log: float,
                 dur_min_ms: float, dur_max_ms: float, gain_median: float,
                 gain_sigma_log: float, rng: np.random.Generator, dt_ms: float):
        self.rate = rate_hz
        self.dur_median = dur_median_ms
        self.dur_sigma = dur_sigma_log
        self.dur_min, self.dur_max = dur_min_ms, dur_max_ms
        self.gain_median = gain_median
        self.gain_sigma = gain_sigma_log
        self.rng = rng
        self.dt_s = dt_ms / 1000.0
        self.dt_ms = dt_ms
        self.active = False
        self.remaining_ms = 0.0
        self.duration_ms = 0.0
        self.gain = 1.0
        self.log: List[dict] = []

    def maybe_start(self) -> bool:
        if self.active:
            return False
        if self.rng.random() < self.rate * self.dt_s:
            self.duration_ms = float(np.clip(self.rng.lognormal(np.log(self.dur_median), self.dur_sigma),
                                             self.dur_min, self.dur_max))
            self.gain = float(self.rng.lognormal(np.log(self.gain_median), self.gain_sigma))
            self.remaining_ms = self.duration_ms
            self.active = True
            self.log.append(dict(duration_ms=self.duration_ms, gain=self.gain))
            return True
        return False

    def step(self) -> bool:
        if self.active:
            self.remaining_ms -= self.dt_ms
            if self.remaining_ms <= 0.0:
                self.active = False
        return self.active

    def abort(self):
        self.active = False


# --------------------------------------------------------------------------- #
# 9. Recording and analysis helpers
# --------------------------------------------------------------------------- #
class SpikeRecorder:
    def __init__(self, n_steps: int, n: int):
        self.S = np.zeros((n_steps, n), dtype=bool)

    def write(self, t: int, spikes: np.ndarray):
        self.S[t] = spikes


def smoothed_rate(S: np.ndarray, dt_ms: float, window_ms: float = 50.0) -> np.ndarray:
    """Population mean firing rate (Hz) smoothed with a boxcar."""
    w = max(1, int(window_ms / dt_ms))
    pop = S.mean(axis=1) / (dt_ms / 1000.0)
    k = np.ones(w) / w
    return np.convolve(pop, k, mode="same")


def tuning_curve(S: np.ndarray, variable: np.ndarray, mask: np.ndarray, dt_ms: float,
                 bins: np.ndarray) -> np.ndarray:
    """Firing rate (Hz) of each neuron as a function of a 1-D variable, within mask."""
    idx = np.digitize(variable[mask], bins) - 1
    nb = len(bins) - 1
    occ = np.bincount(idx, minlength=nb)[:nb] * (dt_ms / 1000.0)
    counts = np.zeros((nb, S.shape[1]))
    Sm = S[mask]
    for b in range(nb):
        sel = idx == b
        if sel.any():
            counts[b] = Sm[sel].sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = counts / occ[:, None]
    rates[occ == 0] = np.nan
    return rates


def binned_counts(S: np.ndarray, dt_ms: float, bin_ms: float) -> np.ndarray:
    w = int(bin_ms / dt_ms)
    nb = S.shape[0] // w
    return S[: nb * w].reshape(nb, w, S.shape[1]).sum(axis=1).astype(float)


def covariance_fc(S: np.ndarray, state_trace: np.ndarray, state_value: int,
                  dt_ms: float, bin_ms: float = 20.0, min_frac: float = 0.5) -> np.ndarray:
    """Covariance matrix of binned spike counts restricted to bins mostly in a state."""
    w = int(bin_ms / dt_ms)
    nb = S.shape[0] // w
    counts = binned_counts(S, dt_ms, bin_ms)
    frac = (state_trace[: nb * w].reshape(nb, w) == state_value).mean(axis=1)
    sel = frac >= min_frac
    if sel.sum() < 3:
        return np.full((S.shape[1], S.shape[1]), np.nan)
    return np.cov(counts[sel].T)
