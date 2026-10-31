"""
hds.scenarios -- ready-made systems. Each returns (system, readout, x0, meta).

nested_rings_hybrid   Sim A + a neural guard: when the guard neuron group is
                      sufficiently active, a stochastic jump (hazard) moves
                      the state to the OTHER ring (jump between attractors,
                      same phase) and enters a gain-up regime (field x G) for
                      an exponential dwell, then returns to base gain.

Templates (compose freely; see hds.dynamics):
  two_points_with_saddle    bistable points separated by a saddle
  ring_with_dwell           ring with a low-energy dwell zone off-ring and a
                            channel flowing from the dwell back to the ring
  distorted_rings           nested rings read out through a geometry warp
"""

import numpy as np
from . import dynamics as dy
from . import neurons as nr


def ring_swap(r_in, r_out, r_bar):
    """Jump map: radially move the state onto the other ring, same phase."""
    def jump(X, rng):
        Y = X.copy()
        rho = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2) + 1e-12
        tgt = np.where(rho > r_bar, r_in, r_out)
        Y[:, 0] *= tgt / rho
        Y[:, 1] *= tgt / rho
        return Y
    return jump


def nested_rings_hybrid(n_trials=24, G=3.0, hazard=3.0, dwell=0.5,
                        guard_dir=(1.0, 0.0, 0.0), guard_halfwidth=0.5,
                        n_neur=1200, gain=1.7, median=7.0, warp=None, seed=0):
    rings = dy.NestedRings()
    field = dy.Sum([rings, dy.AuxDecay(3.0)])
    readout = nr.Readout(n=n_neur, gain=gain, median=median, warp=warp,
                         seed=seed + 1)
    grp = readout.group(guard_dir, frac=0.03)
    # threshold: group rate on the INNER ring at +/- halfwidth from the guard
    # direction (so both rings pass through the guard sector)
    phi0 = np.arctan2(guard_dir[1], guard_dir[0])
    probe = np.array([[rings.r_in * np.cos(phi0 + guard_halfwidth),
                       rings.r_in * np.sin(phi0 + guard_halfwidth), 0.0]])
    theta = readout.expected_rates(probe, np.zeros(1), np.zeros(1, int),
                                   None, grp).mean()

    def cond(X, lam):
        return lam(grp).mean(1) > theta

    modes = {'base': dy.Mode('base'),
             'high': dy.Mode('high', field_gain=G, mean_duration=dwell,
                             exit_to='base')}
    guards = [dy.Guard('neural_guard', cond, hazard,
                       ring_swap(rings.r_in, rings.r_out, rings.r_bar),
                       'high', from_modes=('base',), refractory=0.3)]
    system = dy.HybridSystem(field, modes, guards)
    rng = np.random.default_rng(seed)
    rho = np.where(np.arange(n_trials) % 2 == 0, rings.r_out, rings.r_in)
    th = rng.uniform(0, 2 * np.pi, n_trials)
    x0 = np.stack([rho * np.cos(th), rho * np.sin(th),
                   rng.normal(0, .1, n_trials)], 1)
    meta = dict(rings=rings, guard_group=grp, guard_theta=theta, G=G,
                hazard=hazard, dwell=dwell,
                omega={'outer': abs(rings.w_out), 'inner': abs(rings.w_in)})
    return system, readout, x0, meta


# ------------------------------- templates ------------------------------------

def two_points_with_saddle():
    a = dy.PointAttractor((-0.8, 0), k=4)
    b = dy.PointAttractor((0.8, 0), k=4)
    s = dy.Saddle((0, 0), unstable_dir=(1, 0), lam_u=2, lam_s=4)
    field = dy.Sum([dy.Mixture([a, s, b], [(-0.8, 0), (0, 0), (0.8, 0)],
                               width=0.35), dy.AuxDecay()])
    return field


def ring_with_dwell():
    ring = dy.RingAttractor(radius=1.0, omega=3.0, stiffness=8.0)
    dwell = dy.PointAttractor((0, -1.8), k=0.5)          # weak, slow basin
    chan = dy.Channel((0, -1.6), (0, -1.0), speed=0.8, width=0.15)
    mix = dy.Mixture([ring, dwell], [(0, 0), (0, -1.8)], width=0.6)
    field = dy.Sum([dy.SpeedMap(mix, dy.disk_region((0, -1.8), 0.35),
                                factor=0.15), chan, dy.AuxDecay()])
    return field


def distorted_rings(**kw):
    warp = nr.Compose([nr.RadialStretch(0.3, 2), nr.SineWarp(0.15, 3.0)])
    return nested_rings_hybrid(warp=warp, **kw)
