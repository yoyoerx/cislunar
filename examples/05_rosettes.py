"""Rosette orbits.

1. Earth-Moon resonant orbits (p:q): simple ellipses in an inertial frame, but
   rosettes in the rotating frame.
2. Lunar libration/retrograde orbits (NRHO, DRO): simple loops in the rotating
   frame, but rosettes in a Moon-centred inertial frame.

Outputs examples/output/rosettes_resonant.png and rosettes_lunar.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

import cislunar as cl
from cislunar import frames, periodic as P
from cislunar.viz import plot_trajectories_2d, plot_trajectories_3d

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
S = cl.EARTH_MOON

CASES = [(2, 1, 0.55), (3, 1, None), (4, 1, None), (5, 2, 0.3), (1, 2, None), (2, 3, 0.5)]

fig, axes = plt.subplots(2, len(CASES), figsize=(4.2 * len(CASES), 8.6))
for k, (p, q, e) in enumerate(CASES):
    orb = P.resonant(p, q, e)
    tr = orb.trajectory(3000)
    d_e = np.linalg.norm(tr.r - S.pos1, axis=1)
    d_m = np.linalg.norm(tr.r - S.pos2, axis=1)
    print(f"{orb.family:>13}: period {orb.period_days:6.2f} d, perigee {S.km(d_e.min()):9,.0f} km, "
          f"apogee {S.km(d_e.max()):9,.0f} km, closest to Moon {S.km(d_m.min()):9,.0f} km, "
          f"stability {orb.stability_index():8.2f}")
    tr.label = f"{p}:{q}"
    plot_trajectories_2d(tr, "xy", S, origin="earth", ax=axes[0, k], fit="all",
                         show_lpoints=(), title=f"{p}:{q} rotating", mark_start=False)
    ti = frames.trajectory_to_inertial(tr, "earth")
    plot_trajectories_2d(ti, "xy", S, ax=axes[1, k], title=f"{p}:{q} inertial", mark_start=False)
    moon = frames.primary_track_inertial(tr.t, S, "moon", "earth")
    axes[1, k].plot(*S.km(moon[:, :2]).T, color="#9a9993", lw=0.8, ls=":")
    for ax in axes[:, k]:
        leg = ax.get_legend()
        if leg:
            leg.remove()
        ax.locator_params(nbins=4)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
fig.suptitle("Earth-Moon resonant orbits: rosettes in the rotating frame (top), "
             "ellipses in the inertial frame (bottom)", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT / "rosettes_resonant.png", dpi=130)

# Lunar rosettes: NRHO and DRO seen from a Moon-centred inertial frame
nrho, dro = cl.nrho(), cl.dro(70000.0)
t_n = cl.propagate(nrho.state0, 9 * nrho.period, S, n_points=6000, label="9:2 NRHO, 9 revs")
t_d = cl.propagate(dro.state0, 4 * dro.period, S, n_points=4000, label="DRO, 4 revs")
fig = plt.figure(figsize=(15, 6.5))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
plot_trajectories_3d(frames.trajectory_to_inertial(t_n, "moon"), S, ax=ax1,
                     title="9:2 NRHO, Moon-centred inertial (59 days)")
ax2 = fig.add_subplot(1, 2, 2)
plot_trajectories_2d(frames.trajectory_to_inertial(t_d, "moon"), "xy", S, ax=ax2,
                     title="70,000 km DRO, Moon-centred inertial (56 days)", mark_start=False)
fig.tight_layout()
fig.savefig(OUT / "rosettes_lunar.png", dpi=130)
print(f"figures written to {OUT}")
