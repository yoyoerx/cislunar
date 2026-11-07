"""Two-impulse rendezvous from a 70,000 km DRO to the 9:2 L2 southern NRHO.

Scans departure time x time of flight (porkchop), then plots the cheapest
transfer in 2D and 3D. Outputs in examples/output/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cislunar as cl
from cislunar.viz import (interactive_3d, plot_porkchop, plot_trajectories_2d,
                          plot_trajectories_3d)

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
S = cl.EARTH_MOON

dro = cl.dro(70000.0)
nrho = cl.nrho((9, 2), "south")
print(dro.summary())
print(nrho.summary())

t_dep = np.linspace(0.0, S.nd_time_days(14.0), 22)
tofs = np.linspace(S.nd_time_days(2.0), S.nd_time_days(9.0), 12)
print(f"porkchop: {len(t_dep)} x {len(tofs)} transfers ...")
pc = cl.porkchop(dro, nrho, t_dep, tofs, S, keep_transfers=True, refine=3)
ok = np.isfinite(pc.total_mps).mean() * 100
print(f"converged {ok:.0f}% of grid")
best = pc.best
print("best transfer:\n" + best.summary())

plot_porkchop(pc, title="DRO -> 9:2 NRHO rendezvous cost")
plt.savefig(OUT / "dro_nrho_porkchop.png", dpi=150)

arc = best.trajectory()
td = best.t_depart
dro_tr = cl.propagate(dro.state_at(td), td + dro.period, S, t0=td, label="DRO")
nrho_tr = cl.propagate(nrho.state_at(td), td + nrho.period, S, t0=td, label="9:2 NRHO")
arc.label = f"transfer ({best.total_mps:.0f} m/s, {best.tof_days:.1f} d)"

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
plot_trajectories_2d([dro_tr, nrho_tr, arc], "xy", S, origin="moon", ax=axes[0],
                     title="DRO -> NRHO, XY")
plot_trajectories_2d([dro_tr, nrho_tr, arc], "xz", S, origin="moon", ax=axes[1],
                     title="DRO -> NRHO, XZ")
fig.tight_layout()
fig.savefig(OUT / "dro_nrho_transfer_2d.png", dpi=150)
ax = plot_trajectories_3d([dro_tr, nrho_tr, arc], S, origin="moon",
                          title="DRO -> NRHO transfer", show_lpoints=("L2",))
ax.figure.savefig(OUT / "dro_nrho_transfer_3d.png", dpi=150)
try:
    interactive_3d([dro_tr, nrho_tr, arc], str(OUT / "dro_nrho_transfer.html"), S,
                   origin="moon", title="DRO -> NRHO transfer")
except ImportError as e:
    print(e)
print(f"figures written to {OUT}")
