"""Suspension of the bouncing ball, drawn in the style of KRW Figure 1:
flights live in the base plane (h, v); each reset is a faded cylinder arc
leaping from the guard point g = (0, -V) to its reset image r(g) = (0, cV),
shrinking bounce by bounce into the suspended Zeno circle at the origin."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

G, C = 9.81, 0.8
V1   = 4.4          # first impact speed (drop from h ~ 1)
NARC = 7            # number of resets drawn

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=24, azim=-58)

# ---- base plane ------------------------------------------------------------
hh, vv = np.meshgrid(np.linspace(0, 1.15, 2), np.linspace(-5, 4.2, 2))
ax.plot_surface(hh, vv, 0 * hh, color="#90a4ae", alpha=0.12, zorder=0)
ax.plot([0, 0], [-5, 0], [0, 0], color="#263238", lw=3)
ax.text(0.02, -4.9, 0.04, "guard  G = {h=0, v≤0}", color="#263238", fontsize=9)

# ---- flights (parabolas in the base plane) ---------------------------------
# drop from rest at h0 = V1^2/2g, then post-reset flights at launch speeds c*Vk
v0 = np.linspace(0, -V1, 80)
ax.plot((V1**2 - v0**2) / (2 * G), v0, 0 * v0, color="#1565c0", lw=1.8)
Vs = [V1 * C**k for k in range(NARC + 1)]
for k in range(1, NARC + 1):
    u = C * Vs[k - 1]                       # launch speed after reset k
    vf = np.linspace(u, -u, 80)             # up and back down
    ax.plot((u**2 - vf**2) / (2 * G), vf, 0 * vf, color="#1565c0",
            lw=1.6, alpha=max(0.25, 1 - 0.12 * k))

# ---- faded cylinder arcs (the resets) --------------------------------------
def tube(v_from, v_to, amp, rad, alpha):
    s = np.linspace(0, 1, 60)
    th = np.linspace(0, 2 * np.pi, 24)
    S, TH = np.meshgrid(s, th)
    vc = (1 - S) * v_from + S * v_to        # centerline, v coordinate
    zc = amp * np.sin(np.pi * S)            # centerline, height
    # tangent in (v, z); normal frame spans (h, and the in-plane normal)
    dv = (v_to - v_from) * np.ones_like(S)
    dz = amp * np.pi * np.cos(np.pi * S)
    nrm = np.sqrt(dv**2 + dz**2)
    Hs = rad * np.cos(TH)                   # offset in h (out of the plane)
    Vo = vc + rad * np.sin(TH) * (-dz / nrm)
    Zo = zc + rad * np.sin(TH) * (dv / nrm)
    ax.plot_surface(Hs, Vo, Zo, color="#78909c", alpha=alpha,
                    linewidth=0, antialiased=True)
    ax.plot(0 * s, (1 - s) * v_from + s * v_to, amp * np.sin(np.pi * s),
            color="#37474f", lw=0.9, alpha=min(1, alpha + 0.25))

for k in range(NARC):
    V = Vs[k]
    tube(-V, C * V, amp=0.30 + 0.14 * V, rad=0.05 + 0.022 * V,
         alpha=0.28 * max(0.35, 1 - 0.1 * k))

# g and r(g) markers on the first arc
ax.scatter([0, 0], [-V1, C * V1], [0, 0], color="#b71c1c", s=28, zorder=5)
ax.text(0.03, -V1 - 0.15, 0.1, "g", color="#b71c1c", fontsize=12)
ax.text(0.03, C * V1 + 0.1, 0.1, "r(g) = (0, c·|v|)", color="#b71c1c",
        fontsize=10)

# ---- suspended Zeno circle --------------------------------------------------
th = np.linspace(0, 2 * np.pi, 120)
rz = 0.16
ax.plot(0 * th, rz * np.sin(th) * 0.9, 0.18 + rz * np.cos(th),
        color="#d81b60", lw=2.6)
ax.text(0.03, 0.25, 0.46, "suspended Zeno orbit\nConley index (1,1,0)",
        color="#d81b60", fontsize=9)

ax.set_xlabel("h"); ax.set_ylabel("v"); ax.set_zlabel("s (cylinder)")
ax.set_xlim(0, 1.15); ax.set_ylim(-5, 4.4); ax.set_zlim(0, 1.3)
ax.set_box_aspect((0.55, 1.6, 0.55))
ax.set_title("Bouncing ball on the hybrid suspension: flights in the base "
             "plane,\nfaded reset cylinders leap-frogging g → r(g) into the "
             "Zeno circle")
fig.tight_layout()
fig.savefig("/home/claude/hybrid_cmg/suspension_leapfrog.png", dpi=150)
print("saved suspension_leapfrog.png")
