# cislunar

Python tools for designing and visualizing cislunar orbits in 2D and 3D, and for
computing intercepts, rendezvous and RPOD approach sequences. The dynamics model
is the Earth-Moon circular restricted three-body problem (CR3BP).
See [architecture.md](architecture.md) for design and decisions.

**Read online:** [user manual](https://yoyoerx.github.io/cislunar/manual.html) · [servicing base-station case study](https://yoyoerx.github.io/cislunar/case_study_base_stations.html)

## Install

```
pip install -e .                 # numpy, scipy, matplotlib
pip install -e .[interactive]    # + plotly for interactive 3D HTML
pip install -e .[test]           # + pytest
```

## What's in it

| Area | Functions |
|---|---|
| Dynamics | `cr3bp.propagate`, `propagate_state(with_stm=True)`, `jacobi`, `lagrange_points` |
| Orbits | `lyapunov`, `halo` (L1/L2, north/south), `nrho((9, 2))`, `dro`, `periodic.resonant(p, q)` (Earth-Moon resonant "rosette" orbits), `periodic.continue_arclength`, `PeriodicOrbit.stability_index()` |
| Two-body | `twobody.coe2rv / rv2coe / kepler_propagate`, `lambert.lambert` |
| Frames | `frames.rotating_to_inertial`, `inertial_to_rotating`, `to_lvlh`, `from_lvlh`, `lvlh_matrix`, `speed_kms` |
| Intercepts | `targeting.shoot` (CR3BP Lambert), `intercept`, `porkchop` (with `refine=` clean-up passes), `optimize_transfer`, `keplerian_state_fn`, `raan_for_target` |
| RPOD | `relative.plan_waypoints`, `linear_relative_targeting`, `cw_stm`, `cw_two_impulse`, `passive_abort`, `keep_out_violations`, `corridor_violations` |
| Transient orbits | `manifolds.manifold` (stable/unstable tubes), `efolding_time`, `fate` / `fate_map` (escape via L1/L2, impact, capture; lifetime; FTLE, parallel) |
| Flybys & budgets | `targeting.flyby(miss_km=...)`, `Transfer.v_rel_mps`, `navigation.linear_tcm` / `monte_carlo_tcm` (statistical TCM dv), `navigation.stationkeeping_mc` (Floquet-mode SK), `budget.DeltaVBudget` |
| Servicing study | `servicing.default_nodes`, `node_transfer` (phase-free min-dv rendezvous between any two orbits), `earth_transfer`, `coorbital_phasing`, `cost_matrix` (parallel, cached), `campaign` / `break_even` (dv ledger + rocket-equation mass in LEO) |
| Plots | `viz.plot_trajectories_2d / _3d` (optional `color_by="speed"` heat map, `speed_ref`, `speed_scale="log"`), `plot_manifold`, `plot_fate_map`, `plot_zero_velocity`, `plot_porkchop`, `plot_relative_2d / _3d`, `plot_range_history`, `interactive_3d` |

## Quick start

```python
import cislunar as cl
from cislunar.viz import plot_trajectories_3d

S = cl.EARTH_MOON
nrho = cl.nrho((9, 2), "south")        # Gateway reference NRHO
dro = cl.dro(70000.0)
print(nrho.summary())

tr = cl.intercept(dro, nrho, t_depart=0.0, tof=S.nd_time_days(6.0))
print(tr.summary())                      # dv1, dv2 in m/s

plot_trajectories_3d([dro.trajectory(), nrho.trajectory(), tr.trajectory()],
                     origin="moon")
```

RPOD approach in the target's LVLH frame (x = V-bar, y = -H-bar, z = R-bar toward
the Moon):

```python
from cislunar import frames, relative as R
import numpy as np

xt0 = nrho.state_at(0.0)
xc0 = frames.from_lvlh(xt0, np.r_[S.nd_length(np.array([-10.0, 0, 0])), 0, 0, 0], S)
plan = R.plan_waypoints(xt0, xc0, [
    R.Waypoint(6, (-1.0, 0, 0), "1 km"),
    R.Waypoint(10, (-0.05, 0, 0), "50 m"),
    R.Waypoint(11, (0, 0, 0), "dock"),
], S, final_rhodot_mps=(0.03, 0, 0))
print(plan.summary())
```

## Units and frames

Internally everything is **nondimensional** in the **barycentric rotating frame**
(x from Earth toward Moon, z along the orbital angular momentum). Use the
`System` helpers to convert: `S.km()`, `S.mps()`, `S.days()`, `S.nd_length()`,
`S.nd_time_days()`, `S.state_to_dim()`.

## Examples

```
python examples/01_orbit_families.py       # Lyapunov, halo, NRHO, DRO; halo->NRHO family
python examples/02_dro_to_nrho_porkchop.py # porkchop + best transfer
python examples/03_leo_to_nrho.py          # LEO parking orbit -> NRHO (~2 min)
python examples/04_rpod_nrho.py            # waypoint approach, abort & corridor checks
python examples/05_rosettes.py             # resonant rosettes (rotating vs inertial), lunar rosettes
python examples/06_manifolds_and_fates.py  # manifold tubes; fate/lifetime/FTLE maps (~2.5 min, parallel)
python examples/07_flyby_budget.py         # flyby + return, TCM statistics, station-keeping, budget table
python examples/08_servicing_stations.py   # base-station case study: route matrix + 10/100/1000-mission campaigns
python examples/09_orbit_determination.py # orbit determination: one fix -> hypotheses, telescope track, transient fate
```

Figures (PNG) and interactive HTML go to `examples/output/`.

## User manual

`docs/manual.html` is a single self-contained HTML manual (open it in any browser; works
offline). It is written for readers who know orbits from Kerbal Space Program and covers
the three-body concepts, every capability with interactive figures (orbit-family explorer,
maneuver sandbox, clickable porkchop, LEO->NRHO sensitivity, CW playground, RPOD approach
with abort drifts), recipes, troubleshooting, a glossary and an auto-generated API reference.

```
python docs/build_manual.py              # rebuild from cached results (seconds)
python docs/build_manual.py --rebuild    # recompute everything (~10 min)
python docs/build_manual.py --families   # recompute only orbit families
python docs/build_manual.py --porkchop   # recompute only the porkchop
python docs/build_manual.py --rosettes   # recompute only the rosette orbits
python docs/build_manual.py --transient  # recompute manifolds/fates/flyby/TCM/SK data (~8 min)
python docs/tools/check.py docs/manual.html <outdir>      # screenshots + console errors
python docs/tools/interact.py docs/manual.html <outdir>   # exercise controls (needs playwright + Edge)
python docs/tools/check_rosettes.py docs/manual.html <outdir>   # exercise the rosette section
python docs/tools/check_transient.py docs/manual.html <outdir>  # exercise section 11
python docs/tools/check_speed.py docs/manual.html <outdir>      # exercise the speed heat-map option
```

## Case study: servicing base stations

`docs/case_study_base_stations.html` (offline, interactive) compares eight candidate base
stations (LEO and GEO depots, a TESS-like 2:1 translunar orbit, NRHO, DRO, L1/L2 halos, low
lunar orbit) as homes for reusable satellite servicers, against flying each mission directly
from LEO, for clients in LEO, MEO, GEO, translunar, cislunar and lunar orbits and for
campaigns of 10, 100 and 1,000 missions.

```
python docs/build_case_study.py             # rebuild report from cache
python docs/build_case_study.py --rebuild   # recompute (route matrix resumes from docs/_cache/routes.json)
python docs/tools/check_case_study.py docs/case_study_base_stations.html <outdir>
```

## Orbit determination from observations

`cislunar.od` estimates an unknown object's orbit from whatever you have: position
fixes, telescope angles, range / range-rate, a direction of travel, or the angular
rate of a streak. Observers can be the Earth or Moon centre, a ground site on the
rotating Earth, or another spacecraft.

```python
from cislunar import od
cat = od.OrbitCatalog.default()                     # periodic-orbit families, cached in ~/.cache/cislunar
obs = [od.obs_position(0.0, r_km=[-22400, 91300, 0], sigma_km=20, observer="moon"),
       od.obs_velocity_direction(0.0, az_deg=22, el_deg=0, sigma_deg=3, observer="moon")]  # -> DRO, T = 13.9 d
rep = od.determine_orbit(obs, cat)                  # ranked hypotheses
print(rep.table())
best = rep.best                                     # state, covariance, .predict(days), .match (nearest periodic orbit)
fate = od.assess_fate(best)                         # captured / exits via L1 or L2 / impact, from covariance samples
```

If the data fix all six state components, you get a batch least-squares fit with a
covariance. If they don't (for example a single position fix), the periodic-orbit
catalogue supplies hypotheses, and each family is searched for the member that best
fits the data. The answer is then a ranked list that narrows as you add observations.

## Tests

```
python -m pytest
```

Validation includes Lagrange-point equilibria, Jacobi conservation, the STM
vs. finite differences, periodicity of every family, the 9:2 NRHO period and
perilune/apolune radii, Lambert vs. Vallado Example 7-5, the CW STM vs. numerical
integration, LVLH rate vs. finite differences, and waypoint arrival accuracy.

## Limitations

* CR3BP only: no solar gravity, lunar eccentricity or ephemeris. Expect
  differences of tens of m/s compared with high-fidelity tools on long transfers.
* Impulsive manoeuvres only; single-revolution Lambert seeds; single shooting
  (multiple shooting is planned).
* The LVLH frame is built from rotating-frame velocity (see architecture ADR-4).

## License

MIT, see [LICENSE](LICENSE).
