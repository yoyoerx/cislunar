"""RPOD in the 9:2 NRHO: a chaser 20 km behind the target (near apolune)
approaches through V-bar hold points to docking, with safety checks.

Outputs in examples/output/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cislunar as cl
from cislunar import frames, relative as R
from cislunar.viz import plot_range_history, plot_relative_2d, plot_relative_3d

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
S = cl.EARTH_MOON
KEEPOUT = 0.2  # km

nrho = cl.nrho((9, 2), "south")
t0 = 0.0  # apolune
xt0 = nrho.state_at(t0)
# chaser: 20 km behind and 1 km below (+R-bar is toward the Moon), at rest in LVLH
rel0 = np.concatenate((S.nd_length(np.array([-20.0, 0.0, 1.0])), np.zeros(3)))
xc0 = frames.from_lvlh(xt0, rel0, S, "moon")

waypoints = [
    R.Waypoint(8.0, (-5.0, 0.0, 0.0), "5 km"),
    R.Waypoint(14.0, (-1.0, 0.0, 0.0), "1 km"),
    R.Waypoint(18.0, (-0.2, 0.0, 0.0), "200 m"),
    R.Waypoint(20.0, (-0.03, 0.0, 0.0), "30 m"),
    R.Waypoint(21.0, (0.0, 0.0, 0.0), "dock"),
]
plan = R.plan_waypoints(xt0, xc0, waypoints, S, "moon", t0=t0,
                        final_rhodot_mps=(0.03, 0.0, 0.0))
print(plan.summary())

corr = R.corridor_violations(plan.rel, (-1, 0, 0), 15.0, KEEPOUT)
print(f"approach corridor (15 deg half-angle inside {KEEPOUT * 1000:.0f} m): "
      f"{len(corr)} samples outside")
print("passive abort (all later burns missed, 24 h drift):")
aborts = R.passive_abort(plan, duration_hours=24.0)
for a in aborts[:-2]:
    flag = "UNSAFE" if a.min_range_km < KEEPOUT else "ok"
    print(f"  after burn {a.burn_index + 1}: closest approach "
          f"{a.min_range_km * 1000:8.1f} m at +{a.t_min_hours:5.1f} h  {flag}")

# Compare leg 1 against Clohessy-Wiltshire with the local circular mean motion
r_moon = np.linalg.norm(xt0[:3] - S.pos2)
n_equiv = np.sqrt(S.mu / r_moon**3)
d1_cw, _ = R.cw_two_impulse(rel0, S.nd_length(np.array([-5.0, 0, 0])),
                            S.nd_time_seconds(8 * 3600), n_equiv)
print(f"leg 1 departure dv: CR3BP {plan.maneuvers[0].mag_mps:.3f} m/s vs "
      f"CW (local circular) {S.mps(np.linalg.norm(d1_cw)):.3f} m/s")

fig, axes = plt.subplots(2, 1, figsize=(11, 9))
overlay = [(aborts[0].rel, "drift if burn 2 is missed"),
           (aborts[1].rel, "drift if burn 3 is missed")]
plot_relative_2d(plan, ("vbar", "rbar"), ax=axes[0], keepout_km=KEEPOUT,
                 corridor=(15.0, KEEPOUT), extra=overlay,
                 title="NRHO approach: V-bar / R-bar plane")
plot_range_history(plan, ax=axes[1], keepout_km=KEEPOUT)
fig.tight_layout()
fig.savefig(OUT / "rpod_vbar_rbar.png", dpi=150)

near = plan.rel.range_km < 1.2
final = R.RelativeTrajectory(plan.rel.t[near], plan.rel.rel[near], S, "final approach")
fig, ax = plt.subplots(figsize=(9, 5))
plot_relative_2d(final, ("vbar", "rbar"), ax=ax, keepout_km=KEEPOUT,
                 corridor=(15.0, KEEPOUT), units="m", title="Final approach (inside 1.2 km)")
fig.savefig(OUT / "rpod_final.png", dpi=150)

ax = plot_relative_3d(plan, keepout_km=KEEPOUT)
ax.figure.savefig(OUT / "rpod_3d.png", dpi=150)
print(f"figures written to {OUT}")
