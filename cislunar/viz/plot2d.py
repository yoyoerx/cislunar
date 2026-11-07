"""2D matplotlib plots: trajectories, zero-velocity curves, porkchops, RPOD."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .. import cr3bp
from ..constants import EARTH_MOON, System

# Categorical slots, assigned in fixed order (never cycled past 8).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                   "#256abf", "#184f95", "#0d366b"]
# Speed heat map on lines: same blue ramp, but starting at step 250 so the slow
# (light) end still stands out against a light background.
SPEED_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
EARTH_COLOR = "#5f87b3"
MOON_COLOR = "#9a9993"
FORBIDDEN = "#d9d8d3"

_AXES = {"x": 0, "y": 1, "z": 2}


def series_color(i: int) -> str:
    if i >= len(PALETTE):
        raise ValueError("more than 8 series: fold into 'Other' or use small multiples")
    return PALETTE[i]


def style_axes(ax, xlabel="", ylabel="", title=""):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.set_xlabel(xlabel, color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2)
    if title:
        ax.set_title(title, color=TEXT, fontsize=12, loc="left")


def _origin(system: System, origin: str) -> np.ndarray:
    from ..frames import center_position
    return center_position(system, origin)


def _scale(system: System, units: str) -> float:
    return system.lstar if units == "km" else 1.0


def plot_trajectories_2d(trajs, plane: str = "xy", system: System = EARTH_MOON,
                         units: str = "km", origin: str = "barycenter", ax=None,
                         show_primaries: bool = True, show_lpoints=("L1", "L2"),
                         labels=None, title: str = "", mark_start: bool = True,
                         fit: str = "trajectories", color_by: str | None = None,
                         speed_ref: str | None = None, speed_range=None,
                         speed_scale: str = "linear"):
    """Project one or more Trajectory objects onto a coordinate plane.

    plane: 'xy', 'xz' or 'yz'. units: 'km' or 'nd'. origin shifts rotating-frame
    coordinates to 'earth' or 'moon' (ignored for inertial trajectories).
    fit: 'trajectories' frames the view on the trajectories; 'all' also
    includes the primaries.
    color_by: None (one colour per trajectory) or 'speed' (heat map along the
    path, shared colour scale and a colorbar). speed_ref: 'moon', 'earth' or
    'rotating' (default: the origin if it is a body, else 'rotating'); see
    frames.speed_kms. speed_range: (min, max) km/s for the colour scale.
    speed_scale: 'linear' or 'log' (log shows more detail for orbits like the
    NRHO that are slow most of the time and very fast briefly).
    """
    if not isinstance(trajs, (list, tuple)):
        trajs = [trajs]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    i, j = _AXES[plane[0]], _AXES[plane[1]]
    k = _scale(system, units)
    rotating = all(getattr(t, "frame", "rotating") == "rotating" for t in trajs)
    o = _origin(system, origin) if rotating else np.zeros(3)
    pts = []
    sp = _speed_setup(trajs, system, origin, color_by, speed_ref, speed_range, speed_scale)
    for n, tr in enumerate(trajs):
        r = (tr.r - o) * k
        pts.append(r[:, [i, j]])
        lab = labels[n] if labels else (tr.label or f"trajectory {n + 1}")
        if sp is None:
            c = series_color(n)
            ax.plot(r[:, i], r[:, j], color=c, lw=1.6, label=lab)
        else:
            from matplotlib.collections import LineCollection

            seg = np.stack((r[:-1][:, [i, j]], r[1:][:, [i, j]]), axis=1)
            lc = LineCollection(seg, cmap=sp["cmap"], norm=sp["norm"], linewidths=2.2)
            lc.set_array(0.5 * (sp["values"][n][:-1] + sp["values"][n][1:]))
            ax.add_collection(lc)
            c = TEXT_2
            if len(trajs) > 1:  # identity by direct label (colour now encodes speed)
                top = int(np.argmax(r[:, j]))
                ax.annotate(lab, (r[top, i], r[top, j]), xytext=(0, 5), textcoords="offset points",
                            fontsize=8, color=TEXT_2, ha="center", va="bottom")
        if mark_start:
            ax.plot(r[0, i], r[0, j], "o", ms=5, color=c, mec="white", mew=1.0)
    if rotating and show_primaries:
        _draw_primaries_2d(ax, system, o, k, i, j, along_x=(plane == "yz"))
        if show_lpoints and plane != "yz":
            for name, p in cr3bp.lagrange_points(system.mu).items():
                if name in show_lpoints:
                    q = (p - o) * k
                    ax.plot(q[i], q[j], "+", color=TEXT_2, ms=8, mew=1.2)
                    ax.annotate(name, (q[i], q[j]), xytext=(4, 4),
                                textcoords="offset points", color=TEXT_2, fontsize=9)
    elif not rotating and show_primaries:
        center = trajs[0].frame.split("-", 1)[-1]
        rad = system.r1 if center == "earth" else system.r2
        col = EARTH_COLOR if center == "earth" else MOON_COLOR
        ax.add_patch(plt.Circle((0, 0), rad / system.lstar * k, color=col, zorder=3))
    u = "km" if units == "km" else "nd"
    fr = "rotating" if rotating else trajs[0].frame
    style_axes(ax, f"{plane[0]} [{u}]", f"{plane[1]} [{u}]", title)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    if fit == "trajectories":
        P = np.vstack(pts)
        lo, hi = P.min(axis=0), P.max(axis=0)
        pad = 0.08 * max(hi - lo) + 1e-9
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("equal", adjustable="datalim")
    if sp is not None:
        _speed_colorbar(ax, sp)
    elif len(trajs) >= 2 or any(t.label for t in trajs):
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    ax.text(0.99, 0.01, f"{system.name} {fr} frame", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=TEXT_2)
    return ax


def _draw_primaries_2d(ax, system, o, k, i, j, along_x=False):
    for pos, rad, col, name in ((system.pos1, system.r1, EARTH_COLOR, system.name1),
                                (system.pos2, system.r2, MOON_COLOR, system.name2)):
        if along_x and abs(pos[0] - o[0]) > 1e-12 and np.any(o != 0):
            continue  # viewing along x: only the body at the origin is meaningful
        q = (pos - o) * k
        ax.add_patch(plt.Circle((q[i], q[j]), rad / system.lstar * k, color=col, zorder=3))
        ax.annotate(name, (q[i], q[j]), xytext=(6, -12), textcoords="offset points",
                    color=TEXT_2, fontsize=9)


def plot_zero_velocity(ax, jacobi_c: float, system: System = EARTH_MOON,
                       units: str = "km", origin: str = "barycenter",
                       extent=(-1.5, 1.5, -1.5, 1.5), n: int = 600):
    """Shade the forbidden region (2U < C) in the xy plane for Jacobi constant C."""
    xs = np.linspace(extent[0], extent[1], n)
    ys = np.linspace(extent[2], extent[3], n)
    X, Y = np.meshgrid(xs, ys)
    mu = system.mu
    r1 = np.sqrt((X + mu) ** 2 + Y**2)
    r2 = np.sqrt((X - 1 + mu) ** 2 + Y**2)
    U2 = X**2 + Y**2 + 2 * (1 - mu) / r1 + 2 * mu / r2
    o = _origin(system, origin)
    k = _scale(system, units)
    ax.contourf((X - o[0]) * k, (Y - o[1]) * k, U2 - jacobi_c, levels=[-1e9, 0],
                colors=[FORBIDDEN], alpha=0.8, zorder=0)
    ax.contour((X - o[0]) * k, (Y - o[1]) * k, U2 - jacobi_c, levels=[0],
               colors=[TEXT_2], linewidths=0.6, zorder=0)
    return ax


def plot_porkchop(pc, ax=None, total: bool = True, levels=None, title: str = ""):
    """Contour map of transfer cost over departure time and TOF (days)."""
    s = pc.system
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    Z = pc.total_mps if total else pc.dv1_mps
    D, T = np.meshgrid(s.days(pc.t_departs), s.days(pc.tofs))
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    if levels is None:
        finite = Z[np.isfinite(Z)]
        lo, hi = np.nanpercentile(finite, [0, 90]) if finite.size else (0, 1)
        levels = np.linspace(lo, hi, 12)
    cf = ax.contourf(D, T, Z, levels=levels, cmap=cmap, extend="max")
    ax.contour(D, T, Z, levels=levels, colors="white", linewidths=0.5)
    cb = plt.colorbar(cf, ax=ax)
    cb.set_label("total dv [m/s]" if total else "departure dv [m/s]", color=TEXT_2)
    cb.outline.set_edgecolor(GRID)
    if pc.best is not None:
        b = pc.best
        ax.plot(s.days(b.t_depart), b.tof_days, "o", ms=9, color=PALETTE[1],
                mec="white", mew=2)
        ax.annotate(f"min {b.total_mps:.0f} m/s", (s.days(b.t_depart), b.tof_days),
                    xytext=(8, 8), textcoords="offset points", color=TEXT, fontsize=9)
    style_axes(ax, "departure time [days]", "time of flight [days]",
               title or "Transfer cost")
    return ax


def _rel_axes(plane):
    names = {"vbar": 0, "hbar": 1, "rbar": 2}
    lab = {"vbar": "V-bar [km]", "hbar": "-H-bar [km]", "rbar": "R-bar [km]"}
    return names[plane[0]], names[plane[1]], lab[plane[0]], lab[plane[1]]


def plot_relative_2d(plan_or_rel, plane=("vbar", "rbar"), ax=None, keepout_km=None,
                     corridor=None, extra=None, title: str = "", units: str = "km"):
    """LVLH relative-motion plot (target at origin).

    plan_or_rel : RpodPlan or RelativeTrajectory.
    corridor    : (half_angle_deg, range_km) cone about -V-bar.
    extra       : list of (RelativeTrajectory, label) to overlay (e.g. aborts).
    R-bar points toward the primary and is drawn downward (NASA convention).
    """
    from ..relative import RpodPlan

    plan = plan_or_rel if isinstance(plan_or_rel, RpodPlan) else None
    rel = plan.rel if plan else plan_or_rel
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    i, j, li, lj = _rel_axes(plane)
    f = 1000.0 if units == "m" else 1.0
    if units == "m":
        li, lj = li.replace("km", "m"), lj.replace("km", "m")
    n = 0
    if extra:
        for n, (r2, lab) in enumerate(extra, start=1):
            ax.plot(r2.rho_km[:, i] * f, r2.rho_km[:, j] * f, color=series_color(n),
                    lw=1.0, ls="--", label=lab)
    ax.plot(rel.rho_km[:, i] * f, rel.rho_km[:, j] * f, color=PALETTE[0], lw=1.8,
            label=rel.label or "chaser")
    if plan:
        for m in plan.maneuvers:
            k = int(np.argmin(np.abs(rel.t - m.t)))
            p = rel.rho_km[k] * f
            ax.plot(p[i], p[j], "^", ms=8, color=PALETTE[0], mec="white", mew=1.5)
        span = np.ptp(rel.rho_km[:, [i, j]] * f, axis=0).max() + 1e-12
        placed = []
        for wp in plan.waypoints:
            q = np.asarray(wp.rho_km, float) * f
            if not wp.label or any(np.hypot(q[i] - a, q[j] - b) < 0.06 * span
                                   for a, b in placed):
                continue  # skip labels that would collide
            placed.append((q[i], q[j]))
            ax.annotate(wp.label, (q[i], q[j]), xytext=(4, 8),
                        textcoords="offset points", fontsize=8, color=TEXT_2)
    if keepout_km:
        ax.add_patch(plt.Circle((0, 0), keepout_km * f, fill=False, ls=":",
                                color=TEXT_2, lw=1.0))
        ax.plot([], [], ls=":", color=TEXT_2, lw=1.0, label="keep-out sphere")
    if corridor and plane[0] == "vbar":
        ang, rng = corridor
        t = np.radians(ang)
        xs = -rng * np.cos(t) * f
        ax.fill([0, xs, xs], [0, rng * np.sin(t) * f, -rng * np.sin(t) * f],
                color=PALETTE[2], alpha=0.15, lw=0, label="approach corridor")
    ax.plot(0, 0, "s", ms=8, color=TEXT, label="target")
    if plane[1] == "rbar":
        ax.invert_yaxis()
    style_axes(ax, li, lj, title or "Relative motion (target LVLH)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    return ax


def plot_range_history(plan_or_rel, ax=None, keepout_km=None, title: str = ""):
    from ..relative import RpodPlan

    rel = plan_or_rel.rel if isinstance(plan_or_rel, RpodPlan) else plan_or_rel
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.5))
    t = rel.t_hours - rel.t_hours[0]
    ax.semilogy(t, np.maximum(rel.range_km * 1000.0, 0.1), color=PALETTE[0], lw=1.8)
    ax.set_ylim(bottom=0.1)
    if keepout_km:
        ax.axhline(keepout_km * 1000.0, color=TEXT_2, ls=":", lw=1)
        ax.annotate("keep-out", (t[0], keepout_km * 1000.0), xytext=(2, 3),
                    textcoords="offset points", fontsize=8, color=TEXT_2)
    style_axes(ax, "time [h]", "range [m]", title or "Range to target")
    return ax


# ---------------------------------------------------------------------------
# Transient dynamics
# ---------------------------------------------------------------------------

# Fate colours: the three outcomes use the first three categorical slots (the only
# ones that stay colour-blind-safe when every pair can touch, as in a map);
# "captured" is a dark neutral and forbidden cells a light neutral.
FATE_COLORS = {"exit_L1": PALETTE[0], "exit_L2": PALETTE[1], "impact": PALETTE[2],
               "captured": TEXT_2, "forbidden": FORBIDDEN}
FATE_NAMES = {"exit_L1": "leaves via L1 (toward Earth)", "exit_L2": "leaves via L2 (outward)",
              "impact": "impacts Moon/Earth", "captured": "still captured at horizon",
              "forbidden": "forbidden at this energy"}


def plot_manifold(tubes, plane: str = "xy", system: System = EARTH_MOON, origin: str = "moon",
                  units: str = "km", ax=None, show_orbit: bool = True, title: str = "",
                  show_primaries: bool = True):
    """Plot one or more manifolds.ManifoldTube objects, one colour per tube."""
    if not isinstance(tubes, (list, tuple)):
        tubes = [tubes]
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))
    i, j = _AXES[plane[0]], _AXES[plane[1]]
    k = _scale(system, units)
    o = _origin(system, origin)
    pts = []
    for n, tube in enumerate(tubes):
        c = series_color(n)
        for m, tr in enumerate(tube.trajectories):
            r = (tr.r - o) * k
            pts.append(r[:, [i, j]])
            ax.plot(r[:, i], r[:, j], color=c, lw=0.7, alpha=0.75,
                    label=f"{tube.kind} {tube.branch}" if m == 0 else None)
    if show_orbit and tubes:
        orb = tubes[0].orbit.trajectory(800)
        r = (orb.r - o) * k
        ax.plot(r[:, i], r[:, j], color=TEXT, lw=2.0, label=tubes[0].orbit.family or "orbit")
    if show_primaries:
        _draw_primaries_2d(ax, system, o, k, i, j, along_x=(plane == "yz"))
        for name, p in cr3bp.lagrange_points(system.mu).items():
            if name in ("L1", "L2") and plane != "yz":
                q = (p - o) * k
                ax.plot(q[i], q[j], "+", color=TEXT_2, ms=8, mew=1.2)
                ax.annotate(name, (q[i], q[j]), xytext=(4, 4), textcoords="offset points",
                            color=TEXT_2, fontsize=9)
    u = "km" if units == "km" else "nd"
    style_axes(ax, f"{plane[0]} [{u}]", f"{plane[1]} [{u}]", title or "Invariant manifolds")
    P = np.vstack(pts)
    lo, hi = P.min(axis=0), P.max(axis=0)
    pad = 0.05 * max(hi - lo)
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    return ax


def plot_fate_map(fm, kind: str = "fate", ax=None, title: str = ""):
    """Plot a manifolds.FateMap: kind = 'fate' (categorical), 'lifetime' or 'ftle'."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    from ..manifolds import FATES, FORBIDDEN as FB

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.5))
    X = fm.d_km / 1000.0
    Y = fm.jacobi
    if kind == "fate":
        order = ["forbidden", "captured", "exit_L1", "exit_L2", "impact"]
        idx = {FB: 0, FATES.index("captured"): 1, FATES.index("exit_L1"): 2,
               FATES.index("exit_L2"): 3, FATES.index("impact_moon"): 4,
               FATES.index("impact_earth"): 4}
        Z = np.vectorize(idx.get)(fm.code)
        cmap = ListedColormap([FATE_COLORS[o] for o in order])
        ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=-0.5, vmax=4.5, shading="nearest")
        ax.legend(handles=[Patch(color=FATE_COLORS[o], label=FATE_NAMES[o]) for o in order],
                  frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  labelcolor=TEXT)
    else:
        Z = fm.lifetime_days if kind == "lifetime" else fm.ftle / fm.system.days(1.0)
        Z = np.where(fm.code == FB, np.nan, Z)
        cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
        pc = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="nearest")
        cb = plt.colorbar(pc, ax=ax)
        cb.set_label("lifetime [days]" if kind == "lifetime" else "FTLE [1/day]", color=TEXT_2)
        cb.outline.set_edgecolor(GRID)
    ax.axvline(0, color=TEXT_2, lw=0.6, ls=":")
    style_axes(ax, "start distance from Moon along x [1000 km] (negative = Earth side)",
               "Jacobi constant C", title or f"{kind} map ({fm.sense} starts)")
    return ax



# ---------------------------------------------------------------------------
# Speed heat maps (shared by 2D, 3D and interactive plots)
# ---------------------------------------------------------------------------

def trajectory_speed(tr, system: System = EARTH_MOON, reference: str = "moon"):
    """Speed along a Trajectory in km/s. Rotating-frame trajectories use
    frames.speed_kms(reference); inertial trajectories ('inertial-<body>') give
    the speed relative to their own centre body."""
    from ..frames import speed_kms

    if getattr(tr, "frame", "rotating") == "rotating":
        return speed_kms(tr.x, system, reference)
    return system.kms(np.linalg.norm(tr.v, axis=1))


def _speed_setup(trajs, system, origin, color_by, speed_ref, speed_range,
                 speed_scale: str = "linear"):
    if color_by is None:
        return None
    if color_by != "speed":
        raise ValueError("color_by must be None or 'speed'")
    if speed_scale not in ("linear", "log"):
        raise ValueError("speed_scale must be 'linear' or 'log'")
    from matplotlib.colors import LogNorm, Normalize

    if speed_ref is None:
        speed_ref = origin if origin in ("earth", "moon") else "rotating"
    vals = [trajectory_speed(t, system, speed_ref) for t in trajs]
    lo, hi = speed_range or (min(v.min() for v in vals), max(v.max() for v in vals))
    inertial = getattr(trajs[0], "frame", "rotating") != "rotating"
    if inertial:
        label = f"speed relative to {trajs[0].frame.split('-', 1)[-1].capitalize()} [km/s]"
    elif speed_ref == "rotating":
        label = "speed in the rotating frame [km/s]"
    else:
        label = f"speed relative to {speed_ref.capitalize()} [km/s]"
    if speed_scale == "log":
        lo = max(lo, 1e-3)
        label += " (log scale)"
    norm = LogNorm(lo, hi) if speed_scale == "log" else Normalize(lo, hi)
    return {"values": vals, "norm": norm, "label": label, "ref": speed_ref, "log": speed_scale == "log",
            "cmap": LinearSegmentedColormap.from_list("speed", SPEED_RAMP)}


def _speed_ticks(lo, hi):
    """Round tick values for a log speed scale; at least three."""
    cands = (0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10)
    ticks = [x for x in cands if lo <= x <= hi]
    if len(ticks) < 3:
        ticks = [float(f"{x:.2g}") for x in (lo, np.sqrt(lo * hi), hi)]
    return ticks


def _speed_colorbar(ax, sp):
    from matplotlib.cm import ScalarMappable

    sm = ScalarMappable(norm=sp["norm"], cmap=sp["cmap"])
    cb = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    if sp.get("log"):
        lo, hi = sp["norm"].vmin, sp["norm"].vmax
        ticks = _speed_ticks(lo, hi)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{x:g}" for x in ticks])
        cb.ax.minorticks_off()
    cb.set_label(sp["label"], color=TEXT_2)
    cb.outline.set_edgecolor(GRID)
    cb.ax.tick_params(colors=TEXT_2, labelsize=8)
    return cb
