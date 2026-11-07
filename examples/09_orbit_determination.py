"""Orbit determination: from sparse observations of an unknown object to its orbit.

Three scenarios, each simulated from a known "truth" so the answer can be checked:

A. One position fix (e.g. a radar return) of an object near the Moon. Several
   periodic orbits pass through that point, so the answer is a ranked list of
   hypotheses. Adding a direction of travel (from an image) settles it.
B. A ground telescope tracks an object for two days (angles + Doppler range-rate).
   The data determine the orbit on their own: estimate, covariance, prediction
   with growing uncertainty, and the matching periodic orbit.
C. A spacecraft in the NRHO images an object that has drifted off an L2 halo
   orbit (a transient). The fit identifies the halo it came from, the Delta-v to
   put it back, and a Monte Carlo of the covariance says when it will leave.

The first run builds the periodic-orbit catalogue (~30 s on 6 cores) and caches it
in ~/.cache/cislunar. Output: examples/output/orbit_determination.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cislunar as cl
from cislunar import manifolds as M, od, periodic as P
from cislunar.viz.plot2d import GRID, MOON_COLOR, TEXT_2, series_color, style_axes

OUT = Path(__file__).parent / "output"
S = cl.EARTH_MOON


def moon_xy(x):
    r = S.km(np.atleast_2d(x)[:, :3] - S.pos2)
    return r[:, 0] / 1000, r[:, 1] / 1000


def draw_moon(ax):
    ax.add_patch(plt.Circle((0, 0), S.r2 / 1000, color=MOON_COLOR, zorder=3))


def scenario_a(cat):
    print("\n=== A. One position fix, then a direction of travel ===")
    d = P.dro(70000.0)
    truth = d.state_at(0.2 * d.period)
    fix = od.simulate([od.obs_position(0.0, sigma_km=20.0, observer="moon")], truth, rng=3)
    rep1 = od.determine_orbit(fix, cat, n_hypotheses=8)
    print("[position only]\n" + rep1.table())
    more = fix + od.simulate([od.obs_velocity_direction(0.0, sigma_deg=3.0, observer="moon")], truth, rng=4)
    rep2 = od.determine_orbit(more, cat, n_hypotheses=8)
    print("[position + direction of travel]\n" + rep2.table())
    return truth, rep1, rep2


def scenario_b():
    print("\n=== B. Ground telescope, two nights of angles + range-rate ===")
    truth = P.nrho().state_at(1.0)
    site = od.Observer.ground_site(32.0, -110.0)  # a site in Arizona
    tmpl = []
    for td in np.arange(0.0, 2.0, 0.25):
        tmpl += [od.obs_angles(td, sigma_arcsec=1.0, observer=site),
                 od.obs_range_rate(td, sigma_mps=0.5, observer=site)]
    obs = od.simulate(tmpl, truth, rng=1)
    rep = od.determine_orbit(obs, od.OrbitCatalog.default())
    print(rep.table())
    b = rep.best
    err = b.x - truth
    print(f"  true error {S.km(np.linalg.norm(err[:3])):.1f} km, {S.mps(np.linalg.norm(err[3:])):.2f} m/s "
          f"(Mahalanobis^2 {err @ np.linalg.solve(b.P, err):.1f}; 6 dof, so ~6 expected)")
    days = np.linspace(0.0, 30.0, 61)
    _, sig = b.predict(days)
    for t in (2, 7, 14, 30):
        print(f"  predicted 1-sigma position {t:2d} days out: {np.interp(t, days, sig):8.1f} km")
    return b, days, sig


def scenario_c(cat):
    print("\n=== C. A transient drifting off an L2 halo, imaged from the NRHO ===")
    h = P.halo(2, 20000.0, "south")
    (_, vu), _ = M.eigen_directions(h)
    truth = h.state0 + S.nd_length(200.0) * vu  # 200 km along the unstable direction
    observer = od.Observer.on_orbit(P.nrho(), 0.0, "NRHO observer")
    tmpl = [od.obs_angles(td, sigma_arcsec=5.0, observer=observer, frame="inertial")
            for td in np.arange(0.0, 3.01, 0.5)]
    tmpl += [od.obs_range(td, sigma_km=0.5, observer=observer) for td in (0.0, 1.5, 3.0)]
    rep = od.determine_orbit(od.simulate(tmpl, truth, rng=5), cat)
    print(rep.table())
    b = rep.best
    fa = od.assess_fate(b, n_samples=64, horizon_days=90.0)
    print("  fate of 64 covariance samples over 90 days: "
          + ", ".join(f"{k} {v:.0%}" for k, v in fa["p"].items() if v)
          + "; median time " + ", ".join(f"{k} {v:.1f} d" for k, v in fa["median_days"].items()))
    tf, _ = M.fate(truth, S.nd_time_days(90.0), S)
    print(f"  (truth: {tf.kind} after {S.days(tf.t):.1f} d)")
    print(f"  to keep it on the halo instead: ~{b.match.dv_mps:.1f} m/s at the closest point "
          f"(offset {b.match.dr_km:.0f} km), before the instability grows (e-folding "
          f"{S.days(M.efolding_time(b.match.orbit)):.1f} d)")
    return b, fa


def main():
    OUT.mkdir(exist_ok=True)
    cat = od.OrbitCatalog.default(workers=6, verbose=True)
    print(f"catalogue: {len(cat)} periodic orbits in {len(cat.families)} families")
    truth_a, rep1, rep2 = scenario_a(cat)
    b_b, days, sig = scenario_b()
    b_c, fa = scenario_c(cat)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.3))
    ax = axes[0]
    style_axes(ax, "x from Moon [1000 km]", "y from Moon [1000 km]",
               "A. One fix: which orbit? (solid = still plausible\nafter adding direction of travel)")
    final = {h.match.family for h in rep2.plausible()}
    for i, h in enumerate(rep1.plausible()):
        tr = h.match.orbit.trajectory(1500)
        x, y = moon_xy(tr.x)
        ls = "-" if h.match.family in final else "--"
        ax.plot(x, y, ls, color=series_color(i), lw=1.6,
                label=f"{h.match.family} (weight {h.weight:.2f} -> "
                      f"{'kept' if h.match.family in final else 'ruled out'})")
    fx, fy = moon_xy(truth_a)
    ax.plot(fx, fy, "o", color="#0b0b0b", ms=7, zorder=5, label="position fix")
    draw_moon(ax)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)

    ax = axes[1]
    style_axes(ax, "days after first observation", "1-sigma position uncertainty [km]",
               "B. Telescope track: prediction uncertainty")
    ax.semilogy(days, sig, color=series_color(0), lw=2)
    ax.axvspan(0, 2, color=GRID, alpha=0.6, lw=0)
    ax.text(1.0, sig.max() * 0.6, "tracking\narc", ha="center", color=TEXT_2, fontsize=9)
    ax.text(days[-1], sig[-1], f"  {sig[-1]:,.0f} km", va="center", color=TEXT_2, fontsize=9)
    ax.set_xlim(0, 33)
    peaks = [k for k in range(1, len(sig) - 1) if sig[k] > sig[k - 1] and sig[k] > sig[k + 1]
             and sig[k] > 1.5 * sig[k + 1]]
    for k in peaks[:3]:
        ax.annotate("perilune pass", (days[k], sig[k]), textcoords="offset points", xytext=(6, 4),
                    color=TEXT_2, fontsize=8)

    ax = axes[2]
    style_axes(ax, "x from Moon [1000 km]", "y from Moon [1000 km]",
               f"C. Transient: leaves via L1 after ~{fa['median_days'].get('exit_L1', float('nan')):.0f} d\n"
               f"(24 covariance samples, 1-sigma {b_c.sigma_pos_km():.1f} km: they overlap)")
    H = S.nd_time_days(40.0)
    for k, x0 in enumerate(b_c.samples(24, rng=2)):
        _, tr = M.fate(x0, H, S, n_points=1500)
        x, y = moon_xy(tr.x)
        ax.plot(x, y, color=series_color(1), lw=0.6, alpha=0.5, label="fitted state, sampled" if k == 0 else None)
    x, y = moon_xy(b_c.match.orbit.trajectory(800).x)
    ax.plot(x, y, color=series_color(0), lw=2, label=f"nearest periodic: {b_c.match.family}")
    draw_moon(ax)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "orbit_determination.png", dpi=150)
    print(f"\nwrote {OUT / 'orbit_determination.png'}")


if __name__ == "__main__":
    main()
