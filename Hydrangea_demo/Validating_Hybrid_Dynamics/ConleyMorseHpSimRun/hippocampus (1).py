"""
hippocampus.py
--------------
State-dependent model of a CA1-like ring of pyramidal cells and three
interneuron classes (PV basket, O-LM, chandelier) on a square track, built from
the generic components in neurosim.py.

Construction order (to avoid scoping problems with the self-referential
"sequence" stimulus):
  1. config -> RNG
  2. maze geometry and the position Stimulus (dim 2, evidence 1.0)
  3. macro-state machine (WALK / STILL) and the SWR event generator
  4. population definitions (tuning, background StateParams) -> LIFPopulation
  5. projections and gating traces (inhibitory shaping of place tuning)
  6. TrajectoryMemory: records WALK spike order; later yields a DriveStimulus
  7. simulation loop; replay drive is built from memory at SWR onset
  8. analysis and plots

Run:  python hippocampus.py [config.json]
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neurosim import (BackgroundInput, DriveStimulus, GaussianTuning, LIFPopulation,
                      Oscillator, PoissonEvents, Projection, SpikeRecorder, SpikeTrace,
                      StateMachine, StateParams, Stimulus, covariance_fc, smoothed_rate,
                      tuning_curve)

WALK, STILL, SWR = 0, 1, 2
STATE_LABEL = {WALK: "WALK", STILL: "STILL", SWR: "SWR"}
POP_ORDER = ["pyramidal", "basket_pv", "olm", "chandelier"]
POP_SHORT = {"pyramidal": "Pyr", "basket_pv": "PV", "olm": "O-LM", "chandelier": "Chand"}


# --------------------------------------------------------------------------- #
# 2. Maze geometry
# --------------------------------------------------------------------------- #
class SquareMaze:
    """Square track parameterised by arc length s in [0, 4*side). The track is a ring."""

    def __init__(self, side_m: float):
        self.side = side_m
        self.perimeter = 4.0 * side_m

    def xy(self, s):
        s = np.mod(np.asarray(s, float), self.perimeter)
        edge = np.floor(s / self.side).astype(int)
        u = s - edge * self.side
        x = np.select([edge == 0, edge == 1, edge == 2, edge == 3],
                      [u, self.side, self.side - u, 0.0])
        y = np.select([edge == 0, edge == 1, edge == 2, edge == 3],
                      [0.0, u, self.side, self.side - u])
        return np.stack([x, y], axis=-1)


# --------------------------------------------------------------------------- #
# 6. Trajectory memory (sequence of neural firing during walking)
# --------------------------------------------------------------------------- #
class TrajectoryMemory:
    """Stores the last `size` walking sequences. Each entry holds the pyramidal
    cells ordered by first spike time within a time window, the ring positions
    visited, and the direction of travel."""

    def __init__(self, n_pyr: int, size: int, window_s: float, min_cells: int,
                 recency_tau: float, rng: np.random.Generator):
        self.n = n_pyr
        self.size = size
        self.window_s = window_s
        self.min_cells = min_cells
        self.tau = recency_tau
        self.rng = rng
        self.entries: List[dict] = []
        self.reset_window()

    def reset_window(self):
        self.first = np.full(self.n, np.nan)
        self.window_t = 0.0
        self.positions: List[float] = []

    def record(self, t_s: float, spikes: np.ndarray, s_pos: float, direction: int, dt_s: float):
        new = spikes & np.isnan(self.first)
        self.first[new] = t_s
        self.positions.append(s_pos)
        self.window_t += dt_s
        if self.window_t >= self.window_s:
            self.close_window(direction)

    def close_window(self, direction: int):
        cells = np.where(~np.isnan(self.first))[0]
        if len(cells) >= self.min_cells:
            order = cells[np.argsort(self.first[cells])]
            self.entries.append(dict(cells=order, positions=np.array(self.positions),
                                     direction=direction))
            self.entries = self.entries[-self.size:]
        self.reset_window()

    def select(self) -> Optional[dict]:
        """Pick an entry with probability decaying exponentially with age
        (index 0 = most recent)."""
        if not self.entries:
            return None
        ages = np.arange(len(self.entries))[::-1]  # most recent -> age 0
        w = np.exp(-ages / self.tau)
        w /= w.sum()
        k = self.rng.choice(len(self.entries), p=w)
        return self.entries[k]


class ReplaySchedule:
    """Builds the per-step drive for pyramidal cells during one SWR from a memory
    entry, and feeds it through a DriveStimulus."""

    def __init__(self, n_pyr: int, cfg: dict, memory: TrajectoryMemory, rng: np.random.Generator,
                 dt_ms: float):
        self.n = n_pyr
        self.cfg = cfg
        self.memory = memory
        self.rng = rng
        self.dt_ms = dt_ms
        self.stim = DriveStimulus("replay", n_pyr, evidence=1.0)
        self.schedule = None
        self.step_idx = 0
        self.log: List[dict] = []

    def build(self, duration_ms: float, gain: float):
        n_steps = int(duration_ms / self.dt_ms)
        sched = np.zeros((n_steps, self.n))
        pulse = max(1, int(self.cfg["pulse_ms"] / self.dt_ms))
        entry = self.memory.select()
        kind = "random" if entry is not None else "random_empty_memory"
        reversed_ = False
        if entry is not None and self.rng.random() >= self.cfg["p_random"]:
            cells = entry["cells"]
            k = max(3, int(round(len(cells) * self.cfg["subsample_fraction"])))
            keep = np.sort(self.rng.choice(len(cells), size=k, replace=False))
            cells = cells[keep]
            reversed_ = self.rng.random() < self.cfg["p_reverse"]
            if reversed_:
                cells = cells[::-1]
            onsets = np.linspace(0, max(0, n_steps - pulse), len(cells)).astype(int)
            for c, o in zip(cells, onsets):
                sched[o:o + pulse, c] = 1.0
            kind = "sequence"
        else:
            k = max(1, int(self.n * self.cfg["random_fraction"]))
            cells = self.rng.choice(self.n, size=k, replace=False)
            onsets = self.rng.integers(0, max(1, n_steps - pulse), size=k)
            for c, o in zip(cells, onsets):
                sched[o:o + pulse, c] = 1.0
        self.schedule = sched * self.cfg["drive"]
        self.stim.evidence = float(np.clip(gain, 0.0, 1.0)) if gain <= 1.0 else 1.0
        self.extra_gain = max(1.0, gain)   # gain above 1 enters the ripple multiplicative gain
        self.step_idx = 0
        self.log.append(dict(kind=kind, reversed=reversed_, n_cells=len(cells),
                             duration_ms=duration_ms, gain=gain))

    def step(self) -> np.ndarray:
        if self.schedule is None or self.step_idx >= len(self.schedule):
            self.stim.set(np.zeros(self.n))
        else:
            self.stim.set(self.schedule[self.step_idx])
            self.step_idx += 1
        return self.stim.drive()

    def clear(self):
        self.schedule = None
        self.stim.set(np.zeros(self.n))


# --------------------------------------------------------------------------- #
# 4. Population construction
# --------------------------------------------------------------------------- #
def build_population(name: str, pcfg: dict, maze: SquareMaze, rng: np.random.Generator,
                     dt_ms: float, ring_order: bool) -> LIFPopulation:
    n = pcfg["n"]
    tcfg = pcfg["tuning"]
    if ring_order:
        s_centers = (np.arange(n) + 0.5) / n * maze.perimeter
        s_centers += rng.normal(0.0, tcfg.get("center_jitter_m", 0.0), n)
    else:
        s_centers = rng.uniform(0.0, maze.perimeter, n)
    centers = maze.xy(s_centers)
    tuning = GaussianTuning(centers, tcfg["field_sd_m"], tcfg["amp"],
                            tcfg.get("burst_coupling", 0.0))
    tuning.s_centers = np.mod(s_centers, maze.perimeter)
    params = {st: StateParams.from_dict(d) for st, d in pcfg["states"].items()}
    bg = BackgroundInput(params, rng, n, dt_ms)
    pop = LIFPopulation(name, n, dt_ms, rng, tau_m_ms=pcfg["tau_m_ms"], beta=pcfg["beta"],
                        r0_hz=pcfg["r0_hz"], refractory_ms=pcfg["refractory_ms"],
                        background=bg, tuning=tuning)
    return pop


# --------------------------------------------------------------------------- #
# 7. Simulation
# --------------------------------------------------------------------------- #
def run(cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    dt_ms = cfg["dt_ms"]
    dt_s = dt_ms / 1000.0
    n_steps = int(cfg["duration_s"] / dt_s)

    # 2. maze and the decoded variable
    maze = SquareMaze(cfg["maze"]["side_m"])
    position = Stimulus("position", dim=2, evidence=1.0)

    # 3. states and SWR events
    scfg = cfg["states"]
    sm = StateMachine(scfg["names"], scfg["dwell_mean_s"], scfg["transition_matrix"], rng,
                      dt_ms, scfg["dwell_sigma_log"], initial=scfg.get("initial"))
    wcfg = cfg["swr"]
    swr = PoissonEvents(wcfg["rate_hz"], wcfg["duration_median_ms"], wcfg["duration_sigma_log"],
                        wcfg["duration_min_ms"], wcfg["duration_max_ms"], wcfg["gain_median"],
                        wcfg["gain_sigma_log"], rng, dt_ms)
    swr_refractory_ms = 0.0
    ripple_osc = Oscillator(175.0, dt_ms)
    theta_osc = Oscillator(cfg["locomotion"]["theta_base_hz"], dt_ms)

    # 4. populations
    pops = {name: build_population(name, cfg["populations"][name], maze, rng, dt_ms,
                                   ring_order=(name == "pyramidal")) for name in POP_ORDER}
    pyr, pv, olm, ch = (pops[k] for k in POP_ORDER)

    # 5. connectivity
    ccfg = cfg["connectivity"]
    ring = ccfg["pyr_ring_recurrent"]
    P_ring = Projection.ring(pyr.n, ring["sigma_neurons"], ring["weight"], ring["tau_ms"], dt_ms)
    P_pyr_pv = Projection.random(pyr.n, pv.n, **_pw(ccfg["pyr_to_pv"]), rng=rng, dt_ms=dt_ms)
    P_pv_pyr = Projection.random(pv.n, pyr.n, **_pw(ccfg["pv_to_pyr"]), rng=rng, dt_ms=dt_ms)
    P_pyr_olm = Projection.random(pyr.n, olm.n, **_pw(ccfg["pyr_to_olm"]), rng=rng, dt_ms=dt_ms)
    P_pyr_ch = Projection.random(pyr.n, ch.n, **_pw(ccfg["pyr_to_chandelier"]), rng=rng, dt_ms=dt_ms)
    gcfg, vcfg = ccfg["olm_gate"], ccfg["chandelier_veto"]
    olm_trace = SpikeTrace(gcfg["tau_ms"], dt_ms, olm.n)
    ch_trace = SpikeTrace(vcfg["tau_ms"], dt_ms, ch.n)
    olm_norm = gcfg["norm_rate_hz"] * dt_s * gcfg["tau_ms"] / dt_ms     # trace value at norm rate
    ch_norm = vcfg["norm_rate_hz"] * dt_s * vcfg["tau_ms"] / dt_ms

    # 6. memory and replay
    rcfg = wcfg["replay"]
    memory = TrajectoryMemory(pyr.n, rcfg["memory_size"], rcfg["sequence_window_s"],
                              rcfg["min_cells_per_sequence"], rcfg["recency_tau"], rng)
    replay = ReplaySchedule(pyr.n, rcfg, memory, rng, dt_ms)

    # recorders
    rec = {name: SpikeRecorder(n_steps, pops[name].n) for name in POP_ORDER}
    state_trace = np.zeros(n_steps, dtype=np.int8)
    speed_trace = np.zeros(n_steps)
    pos_trace = np.zeros(n_steps)
    theta_trace = np.zeros(n_steps)
    gain_trace = np.zeros(n_steps)
    dir_trace = np.zeros(n_steps, dtype=np.int8)
    example_v = np.zeros((n_steps, 3))      # one example pyramidal cell's v, p_base, p
    ex_cell = pyr.n // 4

    # locomotion variables
    lcfg = cfg["locomotion"]
    s_pos = rng.uniform(0.0, maze.perimeter)
    speed = 0.0
    direction = 1
    v_target = lcfg["max_speed_m_s"]

    macro = sm.state
    for t in range(n_steps):
        t_s = t * dt_s
        macro, changed = sm.step()
        if changed:
            if macro == "WALK":
                direction = int(rng.choice([-1, 1]))
                v_target = lcfg["max_speed_m_s"] * rng.uniform(lcfg["min_target_fraction"], 1.0)
                memory.reset_window()
            else:
                memory.close_window(direction)
                if swr.active:
                    swr.abort()
                    replay.clear()

        # ---- locomotion and position stimulus
        if macro == "WALK":
            if rng.random() < lcfg["target_change_rate_hz"] * dt_s:
                v_target = lcfg["max_speed_m_s"] * rng.uniform(lcfg["min_target_fraction"], 1.0)
            speed += (dt_s / lcfg["ramp_tau_s"]) * (v_target - speed)
            speed += rng.normal(0.0, lcfg["speed_noise_sd"]) * np.sqrt(dt_s)
            speed = float(np.clip(speed, 0.0, lcfg["max_speed_m_s"]))
        else:
            speed += (dt_s / (lcfg["ramp_tau_s"] * 0.4)) * (0.0 - speed)
            if speed < 0.01:
                speed = 0.0
        s_pos = (s_pos + direction * speed * dt_s) % maze.perimeter
        position.set(maze.xy(s_pos))

        # ---- SWR events during stillness
        swr_refractory_ms = max(0.0, swr_refractory_ms - dt_ms)
        if macro == "STILL" and speed == 0.0 and swr_refractory_ms == 0.0:
            if swr.maybe_start():
                replay.build(swr.duration_ms, swr.gain)
                ripple_osc.step(rng.uniform(*wcfg["ripple_freq_range_hz"]))
        if swr.active:
            still_active = swr.step()
            if not still_active:
                replay.clear()
                swr_refractory_ms = wcfg["refractory_ms"]
        in_swr = swr.active
        if in_swr:
            state = "SWR"
            code = SWR
        elif macro == "WALK":
            state = "WALK"
            code = WALK
        else:
            state = "STILL"
            code = STILL

        # ---- oscillatory gains
        theta_f = lcfg["theta_base_hz"] + lcfg["theta_speed_slope_hz_per_m_s"] * (
            speed - 0.5 * lcfg["max_speed_m_s"])
        theta_phase = theta_osc.step(theta_f)
        ripple_phase = ripple_osc.step()

        gains = {}
        for name in POP_ORDER:
            sp = pops[name].background.get(state)
            if state == "WALK":
                pref = sp.osc_phase
                if name == "pyramidal":
                    # phase precession: preferred phase advances with signed distance into the field
                    ds = (s_pos - pyr.tuning.s_centers + maze.perimeter / 2) % maze.perimeter - maze.perimeter / 2
                    pref = sp.osc_phase - cfg["populations"]["pyramidal"]["tuning"]["precession_rad_per_sd"] \
                        * direction * ds / pyr.tuning.widths
                g = theta_osc.gain(sp.osc_depth, pref)
            elif state == "SWR":
                depth = wcfg["ripple_depth"] * (1.0 if name != "olm" else 0.0)
                g = ripple_osc.gain(depth, 0.0) * replay.extra_gain
            else:
                g = np.ones(pops[name].n)
            gains[name] = g

        # ---- inhibitory gating traces (computed from previous-step spikes)
        olm_x = olm_trace.step(olm.spikes) / olm_norm
        ch_x = ch_trace.step(ch.spikes) / ch_norm
        olm_gate = max(0.0, 1.0 - gcfg["strength"] * olm_x)            # scales external (stimulus) drive
        ch_veto = max(vcfg["floor"], 1.0 - vcfg["strength"] * ch_x)      # scales pyramidal firing probability

        # ---- currents
        i_ring = P_ring.step(pyr.spikes)
        i_pv_pyr = P_pv_pyr.step(pv.spikes)
        i_pyr_pv = P_pyr_pv.step(pyr.spikes)
        i_pyr_olm = P_pyr_olm.step(pyr.spikes)
        i_pyr_ch = P_pyr_ch.step(pyr.spikes)

        sp_pyr = pyr.background.get(state)
        # during an SWR the current position keeps the STILL-state weight; stim_gain of the
        # SWR state applies to the replay drive only, so the replayed map is decoupled from
        # where the animal sits
        place_gain = pyr.background.get("STILL").stim_gain if in_swr else sp_pyr.stim_gain
        place_drive = pyr.tuning.drive(position) * place_gain * olm_gate
        replay_drive = replay.step() * (sp_pyr.stim_gain if in_swr else 0.0)
        I_pyr = sp_pyr.baseline + place_drive + replay_drive + i_ring + i_pv_pyr \
            + pyr.background.noise(state)
        burst_factor = pyr.tuning.burst_factor() if state == "WALK" else 1.0

        sp_pv = pv.background.get(state)
        I_pv = sp_pv.baseline + pv.tuning.drive(position) * sp_pv.stim_gain + i_pyr_pv \
            + pv.background.noise(state)
        sp_olm = olm.background.get(state)
        I_olm = sp_olm.baseline + olm.tuning.drive(position) * sp_olm.stim_gain + i_pyr_olm \
            + olm.background.noise(state)
        sp_ch = ch.background.get(state)
        I_ch = sp_ch.baseline + i_pyr_ch + ch.background.noise(state)

        # ---- update populations
        pyr.step(state, I_pyr, gains["pyramidal"] * ch_veto, burst_factor)
        pv.step(state, I_pv, gains["basket_pv"])
        olm.step(state, I_olm, gains["olm"])
        ch.step(state, I_ch, gains["chandelier"])

        # ---- memory of walking sequences
        if macro == "WALK":
            memory.record(t_s, pyr.spikes, s_pos, direction, dt_s)

        # ---- record
        for name in POP_ORDER:
            rec[name].write(t, pops[name].spikes)
        state_trace[t] = code
        speed_trace[t] = speed
        pos_trace[t] = s_pos
        dir_trace[t] = direction
        theta_trace[t] = theta_f if state == "WALK" else np.nan
        gain_trace[t] = gains["pyramidal"].mean() * ch_veto
        example_v[t] = (pyr.ptrace.last["v"][ex_cell], pyr.ptrace.last["p_base"][ex_cell],
                        pyr.ptrace.last["p"][ex_cell])

    if sm.state == "WALK":
        memory.close_window(direction)

    return dict(cfg=cfg, maze=maze, pops=pops, rec=rec, state=state_trace, speed=speed_trace,
                pos=pos_trace, direction=dir_trace, theta=theta_trace, gain=gain_trace, example_v=example_v,
                ex_cell=ex_cell, dwell=sm.dwell_log, swr_log=swr.log, replay_log=replay.log,
                memory=memory, dt_ms=dt_ms)


def _pw(d: dict) -> dict:
    return dict(p_conn=d["p_conn"], weight=d["weight"], tau_ms=d["tau_ms"])


# --------------------------------------------------------------------------- #
# 8. Analysis and plots
# --------------------------------------------------------------------------- #
def analyse(res: dict, out: str):
    os.makedirs(out, exist_ok=True)
    cfg, dt_ms = res["cfg"], res["dt_ms"]
    acfg = cfg["analysis"]
    state, pos, speed = res["state"], res["pos"], res["speed"]
    maze = res["maze"]
    tt = np.arange(len(state)) * dt_ms / 1000.0
    summary = {}

    # ---- mean rates per population per state
    rates = {}
    for name in POP_ORDER:
        S = res["rec"][name].S
        rates[name] = {STATE_LABEL[k]: float(S[state == k].mean() / (dt_ms / 1000.0))
                       for k in (WALK, STILL, SWR)}
    summary["mean_rate_hz"] = rates

    # ---- Figure 1: state traces and population rates
    fig, ax = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
    ax[0].plot(tt, state, lw=0.8, color="k")
    ax[0].set_yticks([0, 1, 2]); ax[0].set_yticklabels(["WALK", "STILL", "SWR"])
    ax[0].set_ylabel("state")
    ax[1].plot(tt, speed, lw=0.8); ax[1].set_ylabel("speed (m/s)")
    ax[1].plot(tt, res["theta"] / 20.0, lw=0.6, color="tab:orange", label="theta f / 20")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[2].plot(tt, pos, lw=0.8); ax[2].set_ylabel("ring position s (m)")
    for i, name in enumerate(POP_ORDER):
        r = smoothed_rate(res["rec"][name].S, dt_ms, acfg["rate_window_ms"])
        a = ax[3] if name == "pyramidal" else ax[4]
        a.plot(tt, r, lw=0.7, label=POP_SHORT[name])
    ax[3].set_ylabel("Pyr mean rate (Hz)"); ax[3].legend(fontsize=8)
    ax[4].set_ylabel("interneuron rate (Hz)"); ax[4].legend(fontsize=8)
    # raster of a subset of pyramidal cells
    S = res["rec"]["pyramidal"].S
    sub = np.arange(0, S.shape[1], 4)
    ts, ids = np.where(S[:, sub])
    ax[5].scatter(ts * dt_ms / 1000.0, ids, s=0.3, color="k")
    ax[5].set_ylabel("Pyr cell (sub)"); ax[5].set_xlabel("time (s)")
    fig.suptitle("State traces and population firing")
    fig.tight_layout(); fig.savefig(f"{out}/fig1_state_traces.png", dpi=130); plt.close(fig)

    # ---- Figure 2: zoom on a walking bout and on SWRs
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    walk_idx = np.where(state == WALK)[0]
    if len(walk_idx):
        t0 = walk_idx[len(walk_idx) // 3]
        sl = slice(t0, t0 + 3000)
        _raster_panel(ax[0, 0], res, sl, "Walking: 3 s (theta phase precession along the ring)")
        ax[1, 0].plot(tt[sl], res["gain"][sl], lw=0.7, color="tab:red", label="mean Pyr gain (theta x veto)")
        ax[1, 0].plot(tt[sl], res["example_v"][sl, 0], lw=0.7, label=f"v, Pyr #{res['ex_cell']}")
        ax[1, 0].plot(tt[sl], res["example_v"][sl, 2] * 10, lw=0.7, label="p x10")
        ax[1, 0].legend(fontsize=8); ax[1, 0].set_xlabel("time (s)")
    swr_idx = np.where(state == SWR)[0]
    if len(swr_idx):
        starts = swr_idx[np.r_[True, np.diff(swr_idx) > 1]]
        t0 = starts[len(starts) // 2] - 200
        sl = slice(max(0, t0), t0 + 1000)
        _raster_panel(ax[0, 1], res, sl, "Stillness with SWRs: 1 s")
        ax[1, 1].plot(tt[sl], res["gain"][sl], lw=0.7, color="tab:red", label="mean Pyr gain")
        for name in POP_ORDER:
            ax[1, 1].plot(tt[sl], smoothed_rate(res["rec"][name].S[sl], dt_ms, 5.0), lw=0.7,
                          label=POP_SHORT[name])
        ax[1, 1].legend(fontsize=8); ax[1, 1].set_xlabel("time (s)"); ax[1, 1].set_ylabel("rate (Hz, 5 ms)")
    fig.tight_layout(); fig.savefig(f"{out}/fig2_zoom.png", dpi=130); plt.close(fig)

    # ---- Figure 3: example tuning curves per cell type per state
    # Columns 1-3: rate versus ring position (defined only where the animal was in that
    # state). Column 4: population-pooled rate versus distance between the animal and each
    # cell's field centre, which is defined in every state and exposes the decoupling of
    # firing from position during stillness and SWRs.
    bins = np.linspace(0, maze.perimeter, acfg["n_position_bins"] + 1)
    bc = 0.5 * (bins[1:] + bins[:-1])
    dbins = np.linspace(0, maze.perimeter / 2, 21)
    dbc = 0.5 * (dbins[1:] + dbins[:-1])
    k_ex = acfg["example_cells_per_type"]
    fig, ax = plt.subplots(len(POP_ORDER), 4, figsize=(18, 11))
    spatial_info = {}
    for i, name in enumerate(POP_ORDER):
        S = res["rec"][name].S
        n = S.shape[1]
        ex = np.linspace(0, n - 1, k_ex + 2)[1:-1].astype(int)
        s_centers = res["pops"][name].tuning.s_centers
        for j, code in enumerate((WALK, STILL, SWR)):
            mask = state == code
            curves = tuning_curve(S, pos, mask, dt_ms, bins)
            style = dict(lw=1.2) if code == WALK else dict(lw=0, marker="o", ms=4)
            for c in ex:
                ax[i, j].plot(bc, curves[:, c], label=f"{POP_SHORT[name]} #{c}", **style)
            ax[i, j].plot(bc, np.nanmean(curves, axis=1), color="k", ls="--", lw=1.0,
                          marker=None if code == WALK else "x", label="pop. mean")
            ax[i, j].set_title(f"{POP_SHORT[name]} - {STATE_LABEL[code]}", fontsize=10)
            ax[i, j].set_xlim(0, maze.perimeter)
            if code == WALK:
                spatial_info[name] = float(np.nanmean(_spatial_info(curves)))
            # pooled distance tuning
            d = np.abs((pos[mask][:, None] - s_centers[None, :] + maze.perimeter / 2)
                       % maze.perimeter - maze.perimeter / 2)           # (T_state, n)
            Sm = S[mask]
            idx = np.digitize(d, dbins) - 1
            occ = np.bincount(idx.ravel(), minlength=len(dbc))[:len(dbc)] * (dt_ms / 1000.0)
            cnt = np.bincount(idx.ravel(), weights=Sm.ravel().astype(float), minlength=len(dbc))[:len(dbc)]
            with np.errstate(divide="ignore", invalid="ignore"):
                pooled = np.where(occ > 0, cnt / occ, np.nan)
            ax[i, 3].plot(dbc, pooled, marker="o", ms=3, label=STATE_LABEL[code])
        ax[i, 3].set_yscale("log"); ax[i, 3].set_title(f"{POP_SHORT[name]} - pooled, by state", fontsize=10)
        ax[i, 3].legend(fontsize=7)
        ax[i, 0].set_ylabel("rate (Hz)")
        ax[i, 0].legend(fontsize=7)
    for j in range(3):
        ax[-1, j].set_xlabel("ring position s (m)")
    ax[-1, 3].set_xlabel("|animal - field centre| along ring (m)")
    fig.suptitle("Firing rate versus position (place tuning) by state")
    fig.tight_layout(); fig.savefig(f"{out}/fig3_tuning_by_state.png", dpi=130); plt.close(fig)
    summary["spatial_info_bits_per_spike_walk"] = spatial_info

    # ---- Figure 4: dwell times and SWR statistics
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    for j, name in enumerate(["WALK", "STILL"]):
        d = [x[1] for x in res["dwell"] if x[0] == name]
        ax[j].hist(d, bins=12, color="tab:blue" if name == "WALK" else "tab:gray")
        ax[j].set_title(f"{name} dwell (n={len(d)}, mean={np.mean(d) if d else 0:.1f} s)")
        ax[j].set_xlabel("s")
    durs = [x["duration_ms"] for x in res["swr_log"]]
    gns = [x["gain"] for x in res["swr_log"]]
    ax[2].hist(durs, bins=15, color="tab:red"); ax[2].set_title(f"SWR duration (n={len(durs)})"); ax[2].set_xlabel("ms")
    ax[3].hist(gns, bins=15, color="tab:orange"); ax[3].set_title("SWR gain (lognormal)"); ax[3].set_xlabel("gain")
    fig.tight_layout(); fig.savefig(f"{out}/fig4_dwell_swr.png", dpi=130); plt.close(fig)
    summary["dwell_mean_s"] = {name: float(np.mean([x[1] for x in res["dwell"] if x[0] == name] or [0]))
                               for name in ["WALK", "STILL"]}
    summary["swr"] = dict(n=len(durs), duration_ms_mean=float(np.mean(durs or [0])),
                          gain_mean=float(np.mean(gns or [0])),
                          kinds={k: int(sum(1 for x in res["replay_log"] if x["kind"] == k))
                                 for k in ("sequence", "random", "random_empty_memory")},
                          n_reversed=int(sum(1 for x in res["replay_log"] if x["reversed"])),
                          sequences_in_memory=len(res["memory"].entries))

    # ---- Figure 5: functional connectivity (covariance) per state
    S_all = np.concatenate([res["rec"][n].S for n in POP_ORDER], axis=1)
    sizes = [res["rec"][n].S.shape[1] for n in POP_ORDER]
    edges = np.cumsum([0] + sizes)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    fc = {}
    for j, code in enumerate((WALK, STILL, SWR)):
        C = covariance_fc(S_all, state, code, dt_ms, acfg["fc_bin_ms"])
        fc[STATE_LABEL[code]] = C
        off = C.copy(); np.fill_diagonal(off, np.nan)
        v = np.nanpercentile(np.abs(off), 98) if np.isfinite(off).any() else 1.0
        im = ax[j].imshow(C, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        for e in edges[1:-1]:
            ax[j].axhline(e - 0.5, color="k", lw=0.6); ax[j].axvline(e - 0.5, color="k", lw=0.6)
        ax[j].set_xticks(0.5 * (edges[1:] + edges[:-1])); ax[j].set_xticklabels([POP_SHORT[n] for n in POP_ORDER])
        ax[j].set_yticks(0.5 * (edges[1:] + edges[:-1])); ax[j].set_yticklabels([POP_SHORT[n] for n in POP_ORDER])
        ax[j].set_title(f"FC (covariance, {acfg['fc_bin_ms']:.0f} ms bins) - {STATE_LABEL[code]}")
        plt.colorbar(im, ax=ax[j], fraction=0.046)
    fig.tight_layout(); fig.savefig(f"{out}/fig5_fc_covariance.png", dpi=130); plt.close(fig)
    np.savez(f"{out}/fc_matrices.npz", **{k: v for k, v in fc.items()}, pop_edges=edges)

    # ---- Figure 6: pyramidal-only FC with ring ordering, and mean rate bars
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
    Sp = res["rec"]["pyramidal"].S
    for j, code in enumerate((WALK, STILL, SWR)):
        C = covariance_fc(Sp, state, code, dt_ms, acfg["fc_bin_ms"])
        off = C.copy(); np.fill_diagonal(off, np.nan)
        v = np.nanpercentile(np.abs(off), 98) if np.isfinite(off).any() else 1.0
        im = ax[j].imshow(C, cmap="RdBu_r", vmin=-v, vmax=v)
        ax[j].set_title(f"Pyr FC (ring order) - {STATE_LABEL[code]}")
        plt.colorbar(im, ax=ax[j], fraction=0.046)
    x = np.arange(len(POP_ORDER))
    for j, code in enumerate((WALK, STILL, SWR)):
        ax[3].bar(x + (j - 1) * 0.27, [rates[n][STATE_LABEL[code]] for n in POP_ORDER], 0.27,
                  label=STATE_LABEL[code])
    ax[3].set_xticks(x); ax[3].set_xticklabels([POP_SHORT[n] for n in POP_ORDER])
    ax[3].set_yscale("log"); ax[3].set_ylabel("mean rate (Hz)"); ax[3].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/fig6_pyr_fc_and_rates.png", dpi=130); plt.close(fig)

    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _raster_panel(ax, res, sl, title):
    dt_ms = res["dt_ms"]
    off = 0
    colors = ["k", "tab:red", "tab:green", "tab:purple"]
    for name, col in zip(POP_ORDER, colors):
        S = res["rec"][name].S[sl]
        ts, ids = np.where(S)
        ax.scatter((sl.start + ts) * dt_ms / 1000.0, ids + off, s=1.5, color=col, label=POP_SHORT[name])
        ax.axhline(off - 0.5, color="gray", lw=0.5)
        off += S.shape[1]
    ax.set_title(title, fontsize=10); ax.set_ylabel("neuron"); ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(sl.start * dt_ms / 1000.0, (sl.stop) * dt_ms / 1000.0)


def _spatial_info(curves: np.ndarray) -> np.ndarray:
    """Skaggs spatial information (bits/spike), uniform occupancy weighting."""
    r = np.nan_to_num(curves)
    p = np.ones(r.shape[0]) / r.shape[0]
    m = (p[:, None] * r).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        si = np.nansum(p[:, None] * r / m * np.log2(r / m), axis=0)
    si[m <= 0] = np.nan
    return si


def _sp(**kw):
    return kw


# Default configuration, used when no config.json is given. Rates were chosen so
# pyramidal cells fire ~0.5 Hz out of field and ~20-40 Hz in field, interneurons
# 10-30 Hz, with WALK/STILL bouts of ~15/10 s and ~1 Hz SWRs during stillness.
DEFAULT_CFG = dict(
    seed=1, dt_ms=1.0, duration_s=900.0, output_dir="output",
    maze=dict(side_m=1.0),
    states=dict(names=["WALK", "STILL"], dwell_mean_s=dict(WALK=15.0, STILL=10.0),
                transition_matrix=dict(WALK=dict(STILL=1.0), STILL=dict(WALK=1.0)),
                dwell_sigma_log=0.3, initial="WALK"),
    swr=dict(rate_hz=1.0, duration_median_ms=80.0, duration_sigma_log=0.3,
             duration_min_ms=40.0, duration_max_ms=200.0, gain_median=1.0, gain_sigma_log=0.25,
             refractory_ms=150.0, ripple_freq_range_hz=[150.0, 200.0], ripple_depth=0.6,
             replay=dict(memory_size=10, sequence_window_s=3.0, min_cells_per_sequence=8,
                         recency_tau=3.0, pulse_ms=10.0, p_random=0.2, subsample_fraction=0.6,
                         p_reverse=0.4, random_fraction=0.15, drive=1.4)),
    locomotion=dict(theta_base_hz=8.0, theta_speed_slope_hz_per_m_s=4.0, max_speed_m_s=0.35,
                    min_target_fraction=0.5, target_change_rate_hz=0.3, ramp_tau_s=0.8,
                    speed_noise_sd=0.05),
    populations=dict(
        pyramidal=dict(n=120, tau_m_ms=15.0, beta=4.0, r0_hz=20.0, refractory_ms=3.0,
                       tuning=dict(field_sd_m=0.15, amp=1.0, center_jitter_m=0.02,
                                   burst_coupling=1.0, precession_rad_per_sd=1.2),
                       states=dict(WALK=_sp(baseline=0.1, noise_sd=0.05, noise_tau_ms=20.0,
                                            stim_gain=1.0, osc_depth=0.6, osc_phase=np.pi,
                                            burst_prob=0.15, burst_n=3, burst_isi_ms=6.0),
                                   STILL=_sp(baseline=0.05, noise_sd=0.05, noise_tau_ms=20.0,
                                             stim_gain=0.3),
                                   SWR=_sp(baseline=0.15, noise_sd=0.05, noise_tau_ms=20.0,
                                           stim_gain=1.0, burst_prob=0.3, burst_n=3,
                                           burst_isi_ms=5.0))),
        basket_pv=dict(n=30, tau_m_ms=8.0, beta=4.0, r0_hz=20.0, refractory_ms=1.5,
                       tuning=dict(field_sd_m=1.0, amp=0.0),
                       states=dict(WALK=_sp(baseline=0.9, noise_sd=0.05, osc_depth=0.5, osc_phase=0.0),
                                   STILL=_sp(baseline=0.75, noise_sd=0.05),
                                   SWR=_sp(baseline=1.0, noise_sd=0.05))),
        olm=dict(n=15, tau_m_ms=20.0, beta=4.0, r0_hz=20.0, refractory_ms=2.0,
                 tuning=dict(field_sd_m=1.0, amp=0.0),
                 states=dict(WALK=_sp(baseline=0.8, noise_sd=0.05, osc_depth=0.5, osc_phase=np.pi / 2),
                             STILL=_sp(baseline=0.7, noise_sd=0.05),
                             SWR=_sp(baseline=0.45, noise_sd=0.05))),
        chandelier=dict(n=10, tau_m_ms=15.0, beta=4.0, r0_hz=20.0, refractory_ms=2.0,
                        tuning=dict(field_sd_m=1.0, amp=0.0),
                        states=dict(WALK=_sp(baseline=0.75, noise_sd=0.05, osc_depth=0.4, osc_phase=3 * np.pi / 2),
                                    STILL=_sp(baseline=0.7, noise_sd=0.05),
                                    SWR=_sp(baseline=0.55, noise_sd=0.05)))),
    connectivity=dict(pyr_ring_recurrent=dict(sigma_neurons=3.0, weight=0.01, tau_ms=10.0),
                      pyr_to_pv=dict(p_conn=0.3, weight=0.02, tau_ms=5.0),
                      pv_to_pyr=dict(p_conn=0.3, weight=-0.015, tau_ms=8.0),
                      pyr_to_olm=dict(p_conn=0.2, weight=0.015, tau_ms=10.0),
                      pyr_to_chandelier=dict(p_conn=0.2, weight=0.015, tau_ms=10.0),
                      olm_gate=dict(tau_ms=50.0, norm_rate_hz=10.0, strength=0.3),
                      chandelier_veto=dict(tau_ms=50.0, norm_rate_hz=10.0, floor=0.3, strength=0.2)),
    analysis=dict(rate_window_ms=50.0, n_position_bins=40, example_cells_per_type=3, fc_bin_ms=25.0),
)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            cfg = json.load(f)
    else:
        cfg = DEFAULT_CFG
    res = run(cfg)
    out_dir = cfg.get("output_dir", "output")
    os.makedirs(out_dir, exist_ok=True)
    # raw dump for downstream analyses (pca_dynamics.py)
    np.savez_compressed(f"{out_dir}/simulation.npz", dt_ms=res["dt_ms"], state=res["state"],
                        speed=res["speed"], pos=res["pos"], direction=res["direction"],
                        pop_names=np.array(POP_ORDER),
                        pop_sizes=np.array([res["rec"][n].S.shape[1] for n in POP_ORDER]),
                        spikes=np.packbits(np.concatenate([res["rec"][n].S for n in POP_ORDER], axis=1), axis=0),
                        n_steps=len(res["state"]))
    summary = analyse(res, out_dir)
    print(json.dumps(summary, indent=2))
