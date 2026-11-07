"""Transient dynamics: manifold tubes, fates, lifetimes and FTLE.

1. Unstable/stable manifolds of L1 and L2 Lyapunov orbits: the free "tubes" in
   and out of the Moon's region through the L1/L2 gates.
2. A fate map: start on the x-axis near the Moon at a given distance and energy
   (Jacobi constant); does the trajectory stay captured, leave via L1 or L2, or
   impact? Plus lifetime and FTLE maps.

Outputs examples/output/manifolds.png, fate_map.png. The fate map runs in
parallel (~2-4 minutes on 8 cores).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cislunar as cl
from cislunar import manifolds as M, periodic as P
from cislunar.viz import plot_fate_map, plot_manifold

OUT = Path(__file__).parent / "output"
S = cl.EARTH_MOON


def main():
    OUT.mkdir(exist_ok=True)
    l1, l2 = P.lyapunov(1, 20000.0), P.lyapunov(2, 20000.0)
    for o in (l1, l2):
        print(f"{o.family}: e-folding time {S.days(M.efolding_time(o)):.2f} days")
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, o in zip(axes, (l1, l2)):
        tubes = [M.manifold(o, "unstable", "interior", n=40, tf_days=30),
                 M.manifold(o, "unstable", "exterior", n=40, tf_days=30)]
        for t in tubes:
            print(f"  {o.family} {t.kind} {t.branch}: {t.fate_counts()}")
        plot_manifold(tubes, "xy", S, origin="moon", ax=ax,
                      title=f"Unstable manifolds of the {o.family} (30 days)")
    fig.tight_layout()
    fig.savefig(OUT / "manifolds.png", dpi=130)

    d = np.linspace(-50000.0, 55000.0, 70)
    C = np.linspace(2.98, 3.20, 45)
    fm = M.fate_map(d, C, "retrograde", horizon_days=60.0, with_ftle=True, ftle_horizon_days=10.0)
    for k in M.FATES:
        print(f"  {k:13s} {fm.fraction(k) * 100:5.1f}% of allowed starts")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    plot_fate_map(fm, "fate", axes[0], "Fate within 60 days (retrograde starts)")
    plot_fate_map(fm, "lifetime", axes[1], "Lifetime near the Moon")
    plot_fate_map(fm, "ftle", axes[2], "FTLE over 10 days (how fast neighbours separate)")
    fig.tight_layout()
    fig.savefig(OUT / "fate_map.png", dpi=130)
    print(f"figures written to {OUT}")


if __name__ == "__main__":  # required on Windows for the parallel fate map
    main()
