"""Compute and plot the main cislunar periodic-orbit families.

Outputs (examples/output/): families_2d.png, families_3d.png,
halo_family_3d.png, lyapunov_zvc.png, families_3d.html (if plotly installed).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cislunar as cl
from cislunar import periodic as P
from cislunar.viz import (interactive_3d, plot_trajectories_2d, plot_trajectories_3d,
                          plot_zero_velocity)

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
S = cl.EARTH_MOON

print("Lagrange points (km from barycentre):")
for name, p in cl.lagrange_points(S.mu).items():
    print(f"  {name}: x = {S.km(p[0]):10.1f}  y = {S.km(p[1]):10.1f}")

orbits = [
    P.lyapunov(1, 20000.0),
    P.halo(1, 15000.0, "north"),
    P.halo(2, 20000.0, "south"),
    P.nrho((9, 2), "south"),
    P.dro(70000.0),
]
for o in orbits:
    print(o.summary())
trajs = [o.trajectory() for o in orbits]

# 2D projections (Moon-centred)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, plane in zip(axes, ("xy", "xz", "yz")):
    plot_trajectories_2d(trajs, plane, S, origin="moon", ax=ax,
                         title=f"{plane.upper()} projection, Moon-centred rotating frame")
fig.tight_layout()
fig.savefig(OUT / "families_2d.png", dpi=150)

# 3D
ax = plot_trajectories_3d(trajs, S, origin="moon", title="Cislunar periodic orbits")
ax.figure.savefig(OUT / "families_3d.png", dpi=150)

# L2 southern halo family, continued through its fold into the NRHO region
# (pseudo-arclength continuation; free variables x0, z0, vy0)
start = P.halo(2, 5000.0, "south")
fam = P.continue_arclength(start, 0.01, stop=lambda o: o.period_days < 6.0,
                           direction=(0.0, 1.0, 0.0))
picks = fam[:: max(1, len(fam) // 7)][:8]
print(f"L2 southern halo family: {len(fam)} members, plotting {len(picks)}")
for o in picks:
    rp, ra = o.perilune_apolune_km()
    o.family = f"T = {o.period_days:.2f} d, perilune {rp:,.0f} km"
ax = plot_trajectories_3d([o.trajectory(600) for o in picks], S, origin="moon",
                          title="L2 southern halo family -> NRHOs", show_lpoints=("L2",))
ax.figure.savefig(OUT / "halo_family_3d.png", dpi=150)

# Zero-velocity curves for the L1 Lyapunov orbit's energy
lyap = orbits[0]
fig, ax = plt.subplots(figsize=(8, 7))
plot_zero_velocity(ax, lyap.jacobi, S, extent=(0.6, 1.3, -0.35, 0.35))
plot_trajectories_2d(lyap.trajectory(), "xy", S, ax=ax,
                     title=f"L1 Lyapunov, forbidden region at C = {lyap.jacobi:.4f}")
fig.savefig(OUT / "lyapunov_zvc.png", dpi=150)

# Speed heat map: the same orbits coloured by speed relative to the Moon (log scale)
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
plot_trajectories_2d(trajs[1:], "xy", S, origin="moon", ax=axes[0], color_by="speed",
                     speed_scale="log", title="Speed relative to the Moon, XY")
plot_trajectories_2d(trajs[1:], "xz", S, origin="moon", ax=axes[1], color_by="speed",
                     speed_scale="log", title="Speed relative to the Moon, XZ")
fig.tight_layout()
fig.savefig(OUT / "families_speed.png", dpi=150)

try:
    interactive_3d(trajs, str(OUT / "families_3d.html"), S, origin="moon",
                   title="Cislunar periodic orbits (Moon-centred rotating frame)")
    print("wrote families_3d.html")
except ImportError as e:
    print(e)
print(f"figures written to {OUT}")
