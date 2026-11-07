"""Direct transfer from a 400 km, 28.5 deg LEO parking orbit to the 9:2 NRHO.

The LEO plane's RAAN is chosen to contain the NRHO arrival direction, then
optimize_transfer scans departure phase x TOF (two-body Lambert) and refines in
the CR3BP. Plotted in the rotating frame and the Earth-centred inertial frame.
Takes ~2 minutes.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cislunar as cl
from cislunar import frames, targeting
from cislunar.constants import R_EARTH
from cislunar.viz import plot_trajectories_2d, plot_trajectories_3d

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
S = cl.EARTH_MOON

nrho = cl.nrho((9, 2), "south")
inc = np.radians(28.5)
tof_nominal = S.nd_time_days(4.5)
arrive = nrho.state_at(tof_nominal)
r_arr_eci = frames.rotating_to_inertial(tof_nominal, arrive, S, "earth")[:3]
raan = targeting.raan_for_target(r_arr_eci, inc)[0]
print(f"LEO: 400 km, i = 28.5 deg, RAAN = {np.degrees(raan):.2f} deg")
leo = targeting.keplerian_state_fn(R_EARTH + 400.0, 0.0, inc, raan, 0.0, 0.0, S)

leo_period = 2 * np.pi * np.sqrt((R_EARTH + 400.0) ** 3 / S.gm1)
tr = cl.optimize_transfer(leo, nrho, (0.0, S.nd_time_seconds(leo_period)),
                          (S.nd_time_days(3.5), S.nd_time_days(5.5)), S, verbose=True)
print(tr.summary())

arc = tr.trajectory(3000)
arc.label = f"transfer: TLI {tr.dv1_mps:.0f} m/s, NRI {tr.dv2_mps:.0f} m/s"
t_end = tr.t_depart + tr.tof
nrho_tr = cl.propagate(nrho.state_at(t_end), t_end + nrho.period, S, t0=t_end,
                       label="9:2 NRHO")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
plot_trajectories_2d([arc, nrho_tr], "xy", S, origin="earth", ax=axes[0],
                     title="LEO -> NRHO, Earth-centred rotating frame", fit="all")
arc_i = frames.trajectory_to_inertial(arc, "earth")
nrho_i = frames.trajectory_to_inertial(nrho_tr, "earth")
moon_t = np.linspace(tr.t_depart, t_end + nrho.period, 400)
moon_i = frames.primary_track_inertial(moon_t, S, "moon", "earth")
plot_trajectories_2d([arc_i, nrho_i], "xy", S, ax=axes[1], fit="all",
                     title="LEO -> NRHO, Earth-centred inertial frame")
axes[1].plot(*(S.km(moon_i[:, :2]).T), color="#9a9993", lw=1, ls="--", label="Moon path")
axes[1].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "leo_nrho_2d.png", dpi=150)
ax = plot_trajectories_3d([arc, nrho_tr], S, origin="earth", fit="all",
                          title="LEO -> NRHO (rotating frame)")
ax.figure.savefig(OUT / "leo_nrho_3d.png", dpi=150)
print(f"figures written to {OUT}")
