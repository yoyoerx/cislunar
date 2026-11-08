# cislunar -- Architecture

Living document. Update status, phases and ADRs as the project evolves.

Status (2026-11-07): v0.1 -- Phases 0-6g complete (incl. servicing case study, orbit determination, Chang'e 5-T1, TESS and rock-or-rocket real-astrometry case studies); user manual (docs/manual.html) with advanced sections on rosette orbits and on transient orbits / flybys / Delta-v budgets.

## 1. Problem statement

Provide a Python toolkit to:

1. **Calculate** cislunar orbits -- Lagrange points, planar Lyapunov orbits, L1/L2
   halo orbits (including Near Rectilinear Halo Orbits, e.g. the Gateway 9:2 NRHO),
   Distant Retrograde Orbits (DROs), and Earth/Moon-centred Keplerian orbits.
2. **Visualize** them in 2D (planar projections, zero-velocity curves, porkchop plots)
   and 3D (static matplotlib and interactive plotly HTML), in rotating or inertial frames.
3. **Calculate intercepts** from one orbit to another (e.g. LEO -> NRHO, DRO -> NRHO)
   and plan **RPOD** (rendezvous, proximity operations and docking) sequences:
   waypoint-based approaches in the target's LVLH frame, relative-motion propagation,
   linear (STM / Clohessy-Wiltshire) targeting and basic safety checks.

Non-goals (for now): high-fidelity ephemeris (SPICE) dynamics, low-thrust, attitude,
high-fidelity sequential navigation filters (an EKF/UKF may follow `od.py`).

## 2. System diagram

```
              +-------------------+
              |   constants.py    |  System (mu, L*, T*, radii)
              +---------+---------+
                        |
   +--------------------+---------------------+
   |                    |                     |
+--v-------+     +------v------+       +------v------+
| cr3bp.py |     | twobody.py  |       | frames.py   |
| EOM, STM |     | Kepler, COE |       | rot<->inert |
| Jacobi,  |     | lambert.py  |       | LVLH, units |
| L-points |     +------+------+       +------+------+
+--+---+---+            |                     |
   |   |                |                     |
   |  +v----------------v--+       +----------v--------+
   |  | periodic.py        |       | relative.py       |
   |  | diff. correction,  |       | LVLH rel. motion, |
   |  | Lyapunov/halo/NRHO |       | CW, waypoint RPOD |
   |  | DRO, continuation  |       | safety checks     |
   |  +---------+----------+       +---------+---------+
   |            |                            |
   |  +---------v----------+                 |
   +--> targeting.py       <-----------------+
      | CR3BP Lambert-like |
      | intercept/rendez., |
      | porkchop scans     |
      +---------+----------+
                |
      +---------v----------+
      | viz/plot2d, plot3d |
      +--------------------+
```

## 3. Components

| Module | Responsibility |
|---|---|
| `constants.py` | `System` dataclass: mass parameter `mu`, characteristic length/time/velocity, primary radii and GM; `EARTH_MOON` preset; unit conversions. |
| `cr3bp.py` | Circular restricted three-body problem in the barycentric rotating frame (nondimensional). Equations of motion, variational equations (state transition matrix), Jacobi constant, Lagrange points, `propagate()` returning a `Trajectory`. |
| `periodic.py` | Symmetric periodic orbits via differential correction at perpendicular x-z plane crossings. Linear Lyapunov guess, Richardson 3rd-order halo guess, DRO guess, natural-parameter continuation (`continue_to`, `continue_family`), pseudo-arclength continuation through folds (`continue_arclength`), period-constrained correction (used for the 9:2 NRHO), monodromy/stability. `PeriodicOrbit.state_at` uses a cached dense-output interpolant. |
| `twobody.py` | Keplerian elements <-> state, universal-variable Kepler propagation. |
| `lambert.py` | Single-revolution universal-variable Lambert solver (two-body). |
| `frames.py` | Rotating <-> inertial transforms centred on barycentre/Earth/Moon, LVLH frame (and its rate) for a target relative to a chosen primary. |
| `targeting.py` | CR3BP intercept: single-shooting Newton (damped, line search, impact detection) on initial velocity to reach a position at a given time, seeded from Earth- or Moon-centred Lambert. Two-impulse rendezvous, porkchop scans, `optimize_transfer` (Lambert scan + CR3BP Nelder-Mead refinement), Keplerian parking-orbit state providers, `raan_for_target`. |
| `relative.py` | RPOD: nonlinear relative motion in LVLH (both spacecraft propagated in CR3BP), STM-based linear targeting, Clohessy-Wiltshire analytic STM/targeting, multi-waypoint approach planner, keep-out sphere / approach-corridor / passive-abort checks. |
| `manifolds.py` | Transient dynamics: stable/unstable eigen-directions of a periodic orbit, invariant-manifold tubes (globalised by transporting the eigenvector with the STM), fate classification (impact, exit through the L1 or L2 gate, still captured) with lifetime, finite-time Lyapunov exponents, and parallel fate/lifetime/FTLE maps over (distance from Moon, Jacobi constant). |
| `navigation.py` | Statistical Delta-v: injection/execution-error covariances, linear TCM analysis (STM-based, sampled for percentiles), nonlinear Monte Carlo of a TCM strategy (linear guidance, nonlinear truth, navigation and execution errors), and station-keeping Monte Carlo for periodic orbits (target-the-reference with a fixed tracking interval). |
| `budget.py` | `DeltaVBudget`: deterministic / statistical / station-keeping / margin line items, built from `Transfer`, `TcmResult` and `StationKeepingResult` objects, with totals and a printable table. |
| `servicing.py` | Servicing-architecture case study: orbit *nodes* (periodic CR3BP orbits, or Keplerian orbits about Earth/Moon with free phase/RAAN and an optional inclination range), phase-free minimum-Delta-v transfers between nodes (Lambert scan over node parameters and TOF -> CR3BP shooting -> Nelder-Mead), analytic two-body transfers between Earth orbits, co-orbital phasing allowances, a parallel cost-matrix builder, and campaign economics (Delta-v ledger and rocket-equation propellant / IMLEO for direct vs station-based architectures). |
| `od.py` | Orbit determination and identification from sparse observations: observation models (position, velocity, angles, range, range-rate, direction of travel, angle rates from streaks) seen from an observer (Earth/Moon centre, ground site, spacecraft) in rotating / inertial / equatorial axes; initial orbit determination (pos+vel, CR3BP Lambert between fixes); weighted batch least squares (Levenberg-Marquardt, STM partials, optional prior) with covariance; a periodic-orbit *catalogue* (Lyapunov, halo/NRHO, DRO families, cached) used to generate hypotheses when the data cannot determine all six state components; nearest periodic orbit with stability index and insertion Delta-v; covariance prediction and fate Monte Carlo. |
| `ephemeris.py` | Real-world dynamics for working with actual observations: JPL Horizons vector tables (Moon, Sun) fetched once and cached as JSON, Hermite-interpolated; time scales (UTC/TT/TDB, JD); Earth orientation (GMST/GAST, IAU-1976 precession, leading IAU-1980 nutation terms) for station positions in J2000; the IAU lunar rotation model (selenographic coordinates); `NBodyDynamics` - Earth-centred J2000 point-mass Earth+Moon+Sun (+ Earth J2) with STM, Moon-impact events; `CR3BPInFrame` - the CR3BP aligned to the real Moon at an epoch, exposed in the same J2000 frame for like-for-like comparison. |
| `astrometry.py` | MPC 80-column and Find_Orb high-precision astrometry parsing, MPC observatory codes (parallax constants) as `od.Observer`s, conversion to `od` angle observations with light-time, Gauss angles-only initial orbit determination, and a parallel range-scan IOD (trial distances at both ends of a short arc, each refined in the full force model) for objects far from Earth where Gauss's two-body assumption fails; `mpc_stations` pulls parallax constants from the MPC list. |
| `viz/plot2d.py` | Matplotlib 2D: trajectories projected on any plane, primaries and L-points, zero-velocity curves, porkchop contours, LVLH relative plots (V-bar / R-bar). |
| `viz/plot3d.py` | Matplotlib 3D with to-scale Earth/Moon spheres and equal aspect; optional plotly interactive HTML export. |
| `docs/build_manual.py` | Builds `docs/manual.html`: runs the toolkit (cached in `docs/_cache/manual_data.json`), embeds results as JSON, plotly.js, example PNGs and an API reference generated from docstrings into `docs/manual_template.html`. |
| `docs/manual_template.html` | The manual's prose, CSS (light/dark tokens from the validated dataviz palette) and JS: a DOPRI5 CR3BP integrator mirroring `cr3bp.eom`, plus nine interactive figures. |
| `docs/tools/*.py` | Playwright (Edge) scripts (`check.py`, `interact.py`, `check_rosettes.py`, `check_transient.py`) that load the manual, report console errors, screenshot figures and exercise every control. |

## 4. Data flows

* **Orbit design**: `System` -> `periodic.halo(...)` -> `PeriodicOrbit` (state0, period,
  Jacobi, stability) -> `.trajectory()` -> `Trajectory` -> `viz`.
* **Intercept**: chaser state + target `PeriodicOrbit` -> `targeting.rendezvous(t_dep, tof)`
  -> `Transfer` (dv1, dv2, arc `Trajectory`) ; grid of these -> `porkchop()` -> plot.
* **RPOD**: target state + chaser state + list of LVLH waypoints ->
  `relative.plan_waypoints()` -> `RpodPlan` (maneuvers in rotating & LVLH, relative
  trajectory in km) -> safety checks -> `viz.plot_relative_*`.

* **Transient orbits**: `PeriodicOrbit` -> `manifolds.manifold(kind, branch)` -> tube of
  `Trajectory` + `Fate`; initial-condition grid -> `fate_map()` -> fate / lifetime / FTLE arrays.
* **Flyby**: chaser + target -> `targeting.flyby(miss_km=...)` -> `FlybyResult` (departure
  Delta-v, relative speed, actual closest approach); porkchops with `rendezvous=False` give
  departure-only cost plus `Transfer.v_rel`.
* **Budget**: `Transfer` (deterministic) + `navigation.linear_tcm` / `monte_carlo_tcm`
  (statistical) + `navigation.stationkeeping_mc` (per year) -> `budget.DeltaVBudget`.

* **Servicing study**: nodes -> `servicing.cost_matrix()` (parallel `node_transfer` /
  analytic) -> per-route Delta-v table -> `servicing.campaign()` for N missions per
  architecture -> `docs/build_case_study.py` -> `docs/case_study_base_stations.html`.

* **Orbit determination**: `Observation` list (`od.obs_*` constructors) -> `od.determine_orbit(obs, catalog)` -> IOD and catalogue hypotheses -> `od.fit` (batch LS) per hypothesis -> `ODReport` ranked by chi-square, each with covariance, nearest periodic orbit (`OrbitMatch`) and `predict()`; `od.assess_fate` samples the covariance for capture / escape / impact.

* **Real astrometry**: MPC lines -> `astrometry.parse` -> `astrometry.to_observations` (station observers, light-time) -> `astrometry.gauss_iod` -> `od.fit(..., dynamics=ephemeris.NBodyDynamics(...))` -> `ODResult` -> `NBodyDynamics.propagate` (impact event) -> `ephemeris.selenographic`. Case study: `docs/build_case_study_od.py` -> `docs/case_study_change5t1.html`.

All internal computation is **nondimensional CR3BP** in the **barycentric rotating
(synodic) frame**: x from Earth toward Moon, z along the system angular momentum.
Conversion to km, km/s, days happens only at the edges (inputs/outputs, plots).

## 5. Tech decisions (ADRs)

* **ADR-1 Python + numpy/scipy/matplotlib.** Available on the dev machine, standard in
  astrodynamics prototyping. Integration with `scipy.integrate.solve_ivp` (DOP853,
  rtol=atol=1e-12 by default).
* **ADR-2 CR3BP as the core dynamics model.** Captures the essential cislunar
  dynamics (libration orbits, NRHO, DRO) with a cheap, analytic STM. Ephemeris
  dynamics can be added later behind the same `propagate` interface.
* **ADR-3 Plotly is optional.** Static plots must work with matplotlib only;
  `interactive_3d` raises a clear ImportError if plotly is absent.
* **ADR-4 LVLH definition.** Orion/NASA convention relative to a chosen primary
  (Moon by default), built from the target's primary-relative state in the
  *rotating* frame: x = V-bar (j x k), y = -H-bar (negative orbit normal),
  z = R-bar (toward the primary). LVLH rates are computed from the analytic time
  derivative of the rotation matrix using the CR3BP acceleration (no sign-convention
  bookkeeping of angular velocity components). Same axis convention is used for CW.
* **ADR-5 Intercepts by single shooting with Lambert seeds.** Robust enough for
  LEO->NRHO and lunar-vicinity transfers of a few days with a damped Newton step;
  multiple shooting is a later phase for long/chaotic transfers.
* **ADR-6 Symmetric differential correction.** All implemented families (Lyapunov,
  halo, NRHO, DRO) are symmetric about the x-z plane, so half-period correction at the
  y = 0 crossing is used (faster and better conditioned than full-period).
* **ADR-7 ASCII-only console output** (Windows cp1252 console).
* **ADR-8 Halo branch naming.** 'north'/'south' is the sign of z at the crossing
  farther from the Moon (maximum out-of-plane excursion), so the Gateway NRHO is
  'south'. Richardson's tau1 = 0 point is on the Moon side for L2, so the sign is
  flipped there.
* **ADR-9 Pseudo-arclength continuation for families.** The L2 halo family folds in
  z0 (around -29,000 km) before becoming the NRHOs; natural-parameter continuation
  stalls there, so arclength continuation on (x0, z0, vy0) is used.
* **ADR-10 Transfer optimisation by cheap scan + local refinement.** A two-body
  Lambert scan over departure x TOF ranks candidates (milliseconds each); only the
  best few are converged in the CR3BP and refined with Nelder-Mead on warm-started
  shooting. Parking orbits (LEO) use analytic Kepler state providers, and the LEO
  RAAN is chosen to contain the arrival direction to avoid large plane changes.
* **ADR-12 Manual as one offline HTML file.** Data is precomputed in Python and embedded;
  plotly.js is inlined (no CDN) so the 7 MB file works offline. Interactivity that needs new
  trajectories (sandbox, porkchop cell, TLI error) re-integrates in the browser with a JS
  DOPRI5 port of the same equations, from states the toolkit computed, so browser and
  library agree to integration tolerance.
* **ADR-13 Porkchop clean-up passes.** Warm-started sweeps can lock onto expensive solution
  branches (a full spurious stripe appeared in the DRO->NRHO grid). `refine=N` re-solves each
  cell from every neighbour's solution and keeps the cheapest.
* **ADR-14 Resonant (rosette) orbits.** Multi-crossing orbits cannot use the first-crossing
  corrector, so `correct_symmetric_period` integrates a fixed half period with T as a free
  variable. `resonant(p, q, e)` seeds it with a Keplerian ellipse (period q/p lunar months,
  perigee on -x). If direct correction fails, a mass homotopy raises the Moon's mass from
  1e-4 to 1 (geometric steps, refined on failure) holding the perigee distance fixed. Results
  whose period is >10% off 2*pi*q are rejected as a different family. Some p:q/e combinations
  (e.g. 4:3, 5:3 at default e) do not survive to the real mass ratio; the error says so.
* **ADR-15 Manifold globalisation.** Eigenvectors of the monodromy matrix at state0 are
  transported to N phases with the STM (v(t) = Phi(t) v0) and the initial step is a fixed
  position displacement eps (km) along the normalised vector. Stable manifolds are
  integrated backward. Branches are named interior (first step toward Earth, -x) and
  exterior. Integration stops at impact.
* **ADR-16 Fates as gate crossings.** A trajectory starting near the Moon is classified by
  the first terminal event within a horizon: Moon/Earth impact, crossing x = x_L1 moving
  Earthward ("exit via L1"), crossing x = x_L2 moving outward ("exit via L2"), else
  "captured". This matches the zero-velocity-curve picture (section 5 of the manual).
  Fate maps run cells in parallel (ProcessPoolExecutor) because each cell is an
  independent long integration.
* **ADR-17 Statistical Delta-v by linear covariance first, Monte Carlo to confirm.**
  TCM costs are linear in the injection error, so percentiles come from sampling the
  STM-derived linear map (cheap, scalable by sigma). The Monte Carlo uses the same linear
  guidance law but nonlinear truth propagation, navigation-knowledge error and execution
  error, to check that linearity holds (LEO->NRHO: linear p99 15.8 m/s vs Monte Carlo
  16.9 m/s with a second TCM and nav errors).
  Station-keeping uses **Floquet-mode control**: at each tracking pass, cancel the
  deviation's component along the unstable mode (left eigenvector of the monodromy
  matrix transported to the current phase) with the minimum-norm burn. A first
  "target-the-reference" law (null position deviation half a period ahead) was
  rejected: its Phi_rv became ill-conditioned at some phases and it diverged for halos
  and for NRHO passes near perilune, and it spent fuel on stable orbits. Passes where
  the burn efficiency |w_v| is below 25% of its best value (NRHO perilune) are skipped.
  Navigation and execution errors are the only noise sources (CR3BP has no unmodelled
  forces), so absolute values are lower bounds; trends versus interval and orbit are
  the meaningful output. Deviation is measured from the reference orbit's path by
  projecting onto the segments between path samples (vertex-only distances produced
  ~100 km artefacts near the NRHO perilune, where samples are ~230 km apart).
* **ADR-18 Flyby targeting.** `flyby()` aims at target position + miss offset, with the
  offset made perpendicular to the encounter relative velocity (re-solved until the
  relative velocity converges), then reports the true closest approach from a dense
  search around the encounter. Relative speed at encounter is a first-class output
  (`Transfer.v_rel`).
* **ADR-19 Speed heat maps.** `color_by="speed"` on the 2D/3D/interactive trajectory plots
  colours paths by `frames.speed_kms` relative to a body (non-rotating, KSP's "orbital
  speed") or in the rotating frame. Speed is a magnitude, so it uses the one-hue
  sequential blue ramp, starting at step 250 (light mode) / stopping at step 600 (dark)
  so the slow end stays visible on lines. With colour spent on speed, trajectories are
  identified by direct labels instead of a legend. A log option exists because NRHO-like
  orbits span ~0.07 to ~1.7 km/s and a linear scale washes most of the orbit out.
  Plotly has no log colour scale, so log10(speed) is coloured with km/s tick labels.
* **ADR-20 Phase-free node transfers.** The CR3BP is autonomous, so a transfer between two
  orbits depends only on where on each orbit it starts/ends and on the TOF, not on absolute
  time. Periodic orbits contribute a phase parameter; Keplerian orbits contribute argument of
  latitude and RAAN (and inclination when a range is allowed, e.g. our own LEO depot). The
  minimum over these parameters is the cost for a servicer that can wait for good phasing;
  a per-leg allowance covers phasing and navigation. Search: quasi-random Lambert scan
  (Earth- and Moon-centred seeds) -> CR3BP shooting on the best seeds -> Nelder-Mead on
  all parameters with warm-started shooting.
* **ADR-21 Reference planes.** CR3BP z is the Moon's orbit normal. Earth orbits are specified
  by inclination to the equator; the equator is taken as inclined eps = 23.4 deg to the
  Moon's orbit plane. GEO is therefore fixed at 23.4 deg relative; for client classes whose
  RAANs are spread (GNSS, SSO) a nominal relative inclination equal to the equatorial one is
  used; for our own LEO depot the plane is chosen freely within [|28.5 - eps|, 28.5 + eps].
  Earth-to-Earth legs use two-body Hohmann transfers with an optimally split plane change
  and equatorial inclination differences (RAANs aligned by waiting for nodal drift).
* **ADR-22 Return legs by time-reversal symmetry.** The CR3BP is symmetric under
  (x, y, z, vx, vy, vz, t) -> (x, -y, z, -vx, vy, -vz, -t); every node family used is mapped
  onto itself, so the minimum-Delta-v return client -> station equals the outbound
  station -> client cost. Spot-checked numerically for representative pairs.
* **ADR-23 Campaign economics in mass, not just Delta-v.** Delta-v sums across vehicles of
  different mass are not comparable, so the case study reports both the Delta-v ledger and
  rocket-equation propellant / IMLEO for a stated mass model (station, reusable servicer,
  servicing kit, Isp, servicer life, optional lunar-sourced propellant).
* **ADR-24 Case-study modelling fixes (from reviewing results).** (a) The direct baseline
  launches Earth-orbit servicers into the client's own plane (launch-site minimum 28.5 deg);
  forcing them through the LEO depot plane made SSO look 50x too expensive. (b) Mixed-fleet
  plans let each client class choose a station *or* direct launch. (c) Lunar propellant is
  priced (lambda x lunar-surface mass added to LEO mass); unpriced, it made a LEO depot fed
  from the Moon look best. (d) Route orientation per pair (LLO as departure end; periodic
  orbit as departure end for Earth orbits) - escape-burn seeds on every Keplerian departure
  made Earth routes ~10x slower. Failed direct searches fall back to the best two-leg route
  (flagged). Co-orbital phasing on small halos uses a drift-rate estimate because their
  family period is nearly constant.
* **ADR-25 Orbit determination by batch least squares in the CR3BP.** Observations are few and sparse (a few
  fixes, angles over a night, a streak), so a batch estimator (Gauss-Newton with Levenberg-Marquardt damping) on the
  epoch state is simpler and more robust than a sequential filter; partials are measurement Jacobians (central finite
  differences of cheap analytic models) times the STM. Covariance = inverse information matrix. Ground-site observers
  rotate with the Earth about an axis tilted 23.44 deg from the Earth-Moon plane normal (node on the inertial x-axis);
  the Moon's 5 deg tilt and 18.6-yr nodal cycle are ignored, consistent with ADR-21.
* **ADR-26 Catalogue hypotheses for under-determined data.** One position fix, or a position with only a direction of
  travel, leaves the state under-determined. Instead of failing, the estimator draws hypotheses from a catalogue of
  periodic orbits (operators put satellites on them) sampled in phase; a weak prior centred on each matching catalogue
  state (default 2,000 km / 50 m/s) closes the problem, and hypotheses are ranked by chi-square. The answer is then a
  ranked *list* of plausible orbits, which narrows as observations are added. Catalogue states between samples use cubic
  Hermite interpolation (derivatives from the EOM). The nearest *exact* periodic orbit is found by re-correcting
  interpolated family members (bounded 1-D search between neighbours). "Stable" means linearly stable (stability index
  nu <= 1); the fate Monte Carlo reports what the fitted (possibly transient) state actually does.
* **ADR-27 Pluggable dynamics in `od`; an ephemeris model for real data.** Real astrometry of a cislunar object cannot be
  fitted in the CR3BP: over 5 days the Sun's tidal acceleration alone displaces a 300,000-km-distant object by ~3,000 km
  (arcminutes on the sky), and the real Moon is up to 21,000 km from its circular CR3BP position. `od.fit` therefore takes
  a `dynamics` object (`propagate_many(x0, t0, times, with_stm)`); the default stays the CR3BP. `NBodyDynamics` works in
  Earth-centred J2000 with the same nondimensional units (L*, T*), so `od`'s scaling, priors and reporting carry over.
  Moon/Sun come from JPL Horizons tables (cached in the repo so builds are reproducible offline); STM partials ignore J2
  (negligible). Solar radiation pressure is an optional cannonball term (C_r A/m, no shadow); the Chang'e case study
  shows it dominates a 6-week lunar-impact forecast yet is unobservable from a 5-day arc. Not modelled: shadow, relativity.
* **ADR-28 Astrometric measurement model.** MPC positions are astrometric (J2000, light-time included, no aberration):
  the model is the geometric topocentric direction to the target at t - rho/c (first-order back-step along the velocity,
  two iterations). Stations come from MPC parallax constants rotated by GAST, nutation (leading terms, ~0.5" error) and
  IAU-1976 precession; UT1 = UTC, no polar motion (< 50 m at the site). Observation noise defaults to 1" per axis; the
  fit reports per-station residuals so outliers can be down-weighted.
* **ADR-29 Case-study data provenance.** Real data lives in `docs/data/<case>/` with source URLs and fetch dates; the
  ground truth (LROC crater) is kept separate from anything used in the fit.
* **ADR-30 Area-to-mass ratio as an identification test.** Solar radiation pressure (C_r A/m) is profiled by refitting the
  orbit on a grid of fixed values and fitting a parabola to chi-square (simpler than adding a 7th estimated parameter, and it
  shows the whole likelihood). Outliers are rejected at 5x the robust scatter (min 3"), re-done at the best value. Hollow
  hardware (~0.01-0.02 m^2/kg) and rock (~3e-5 to 2e-4) differ ~100x, so a months-long arc separates them decisively.
* **ADR-11 Performance.** Pure numpy/scipy; the STM right-hand side is hand-expanded
  (~24 us/call). Trial trajectories that hit a primary are stopped by an event.
  numba is a possible future accelerator (not installed on the dev machine).

## 6. Project structure

```
cislunar/
  architecture.md        this file
  README.md
  pyproject.toml
  cislunar/
    __init__.py
    constants.py
    cr3bp.py
    periodic.py
    twobody.py
    lambert.py
    frames.py
    targeting.py
    relative.py
    viz/
      __init__.py
      plot2d.py
      plot3d.py
  examples/              runnable scripts; write images/HTML to examples/output/
  tests/                 pytest suite
```

## 7. Development phases

- [x] Phase 0 -- architecture, project skeleton
- [x] Phase 1 -- CR3BP core: EOM, STM, Jacobi, Lagrange points, propagation
- [x] Phase 2 -- periodic orbits: Lyapunov, halo (Richardson + correction), NRHO 9:2, DRO, continuation, stability
- [x] Phase 3 -- two-body + Lambert, frames (inertial, LVLH)
- [x] Phase 4 -- CR3BP intercept/rendezvous targeting, porkchop
- [x] Phase 5 -- RPOD: relative motion, CW, waypoint planner, safety checks
- [x] Phase 6 -- 2D/3D visualization, examples, tests (28 tests passing)
- [x] Phase 6b -- HTML user manual with interactive simulations (docs/)
- [x] Phase 6c -- resonant rosette orbits (`periodic.resonant`) + advanced manual section
- [ ] Phase 7 -- multiple shooting; multi-rev Lambert (Izzo); finite burns
- [ ] Phase 7b -- RPOD extensions: true hold points (station-keeping), passively
      safe trajectory design, sensor/visibility constraints, Monte Carlo dispersions
- [ ] Phase 8 -- ephemeris (SPICE / JPL DE) dynamics and CR3BP -> ephemeris transition
- [x] Phase 6d -- transient orbits & Delta-v budgets: manifolds, fate/lifetime/FTLE maps,
      flyby targeting, linear + Monte Carlo TCM, station-keeping Monte Carlo, budget
      roll-up, manual section 11 (44 tests passing)
- [x] Phase 6f -- servicing-station case study (`servicing.py`, docs/case_study_base_stations.html; 55-route matrix, 32 min on 7 cores)
- [x] Phase 6g -- orbit determination & identification from sparse observations (`od.py`)
- [x] Phase 6h -- real-data OD case study: Chang'e 5-T1 stage lunar impact (`ephemeris.py`, `astrometry.py`)
- [x] Phase 6i -- second real-data OD case study: TESS stable 2:1 lunar-resonant orbit vs NASA truth (`docs/case_study_tess.html`)
- [x] Phase 6j -- rock-or-rocket case study (2020 SO vs 2024 PT5), range-scan IOD, MPC station loader
- [ ] Phase 9 -- additional families (axial, vertical, butterfly); quasi-periodic tori

## 8. Open questions

* Should LVLH be defined from inertial (Moon-centred) or rotating-frame velocity?
  Currently rotating (ADR-4); differences are small for close NRHO perilune passes but
  noticeable near apolune.
* Which ephemeris source to adopt (SPICE kernels vs. astropy/jplephem)?
