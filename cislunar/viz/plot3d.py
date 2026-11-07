"""3D plots: matplotlib (static) and plotly (interactive HTML, optional)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .. import cr3bp
from ..constants import EARTH_MOON, System
from .plot2d import (EARTH_COLOR, GRID, MOON_COLOR, PALETTE, SPEED_RAMP, TEXT, TEXT_2,
                     _origin, _scale, _speed_colorbar, _speed_setup, _speed_ticks,
                     series_color)


def _sphere(center, radius, n=24):
    u, v = np.mgrid[0:2 * np.pi:n * 1j, 0:np.pi:(n // 2) * 1j]
    return (center[0] + radius * np.cos(u) * np.sin(v),
            center[1] + radius * np.sin(u) * np.sin(v),
            center[2] + radius * np.cos(v))


def _set_equal_3d(ax, pts):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c = 0.5 * (lo + hi)
    h = 0.5 * (hi - lo).max() * 1.05
    ax.set_xlim(c[0] - h, c[0] + h)
    ax.set_ylim(c[1] - h, c[1] + h)
    ax.set_zlim(c[2] - h, c[2] + h)
    ax.set_box_aspect((1, 1, 1))


def _style_3d(ax, u, title):
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor("white")
        a.pane.set_edgecolor(GRID)
        a._axinfo["grid"]["color"] = GRID
        a.label.set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.set_xlabel(f"x [{u}]")
    ax.set_ylabel(f"y [{u}]")
    ax.set_zlabel(f"z [{u}]")
    if title:
        ax.set_title(title, color=TEXT, fontsize=12, loc="left")


def plot_trajectories_3d(trajs, system: System = EARTH_MOON, units: str = "km",
                         origin: str = "barycenter", ax=None, show_primaries: bool = True,
                         show_lpoints=("L1", "L2"), labels=None, title: str = "",
                         fit: str = "trajectories", color_by: str | None = None,
                         speed_ref: str | None = None, speed_range=None,
                         speed_scale: str = "linear"):
    """3D plot of trajectories with to-scale primaries and equal aspect.

    fit: 'trajectories' (zoom to the trajectories) or 'all' (include primaries).
    color_by='speed' colours each path by speed (see plot_trajectories_2d).
    """
    if not isinstance(trajs, (list, tuple)):
        trajs = [trajs]
    if ax is None:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(projection="3d")
    k = _scale(system, units)
    rotating = all(getattr(t, "frame", "rotating") == "rotating" for t in trajs)
    o = _origin(system, origin) if rotating else np.zeros(3)
    pts = []
    sp = _speed_setup(trajs, system, origin, color_by, speed_ref, speed_range, speed_scale)
    for n, tr in enumerate(trajs):
        r = (tr.r - o) * k
        pts.append(r)
        lab = labels[n] if labels else (tr.label or f"trajectory {n + 1}")
        if sp is None:
            ax.plot(r[:, 0], r[:, 1], r[:, 2], color=series_color(n), lw=1.5, label=lab)
        else:
            from mpl_toolkits.mplot3d.art3d import Line3DCollection

            seg = np.stack((r[:-1], r[1:]), axis=1)
            lc = Line3DCollection(seg, cmap=sp["cmap"], norm=sp["norm"], linewidths=2.0)
            lc.set_array(0.5 * (sp["values"][n][:-1] + sp["values"][n][1:]))
            ax.add_collection3d(lc)
            if len(trajs) > 1:
                ax.text(*r[int(np.argmax(r[:, 2]))], f" {lab}", color=TEXT_2, fontsize=8)
    bodies = []
    if show_primaries:
        if rotating:
            bodies = [(system.pos1, system.r1, EARTH_COLOR), (system.pos2, system.r2, MOON_COLOR)]
        else:
            center = trajs[0].frame.split("-", 1)[-1]
            bodies = [(np.zeros(3) + o, system.r1 if center == "earth" else system.r2,
                       EARTH_COLOR if center == "earth" else MOON_COLOR)]
    for pos, rad, col in bodies:
        X, Y, Z = _sphere((pos - o) * k, rad / system.lstar * k)
        ax.plot_surface(X, Y, Z, color=col, linewidth=0, shade=True, alpha=0.95)
        if fit == "all":
            pts.append(((pos - o) * k)[None, :])
    if rotating and show_lpoints:
        for name, p in cr3bp.lagrange_points(system.mu).items():
            if name in show_lpoints:
                q = (p - o) * k
                ax.scatter(*q, marker="+", color=TEXT_2, s=40)
                ax.text(*q, f" {name}", color=TEXT_2, fontsize=8)
    u = "km" if units == "km" else "nd"
    _style_3d(ax, u, title)
    _set_equal_3d(ax, np.vstack(pts))
    if sp is not None:
        _speed_colorbar(ax, sp)
    elif len(trajs) >= 2 or any(t.label for t in trajs):
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper left")
    return ax


def plot_relative_3d(plan_or_rel, ax=None, keepout_km=None, title: str = ""):
    """3D LVLH relative trajectory (target at origin). Axes: V-bar, -H-bar, R-bar."""
    from ..relative import RpodPlan

    plan = plan_or_rel if isinstance(plan_or_rel, RpodPlan) else None
    rel = plan.rel if plan else plan_or_rel
    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(projection="3d")
    r = rel.rho_km
    ax.plot(r[:, 0], r[:, 1], r[:, 2], color=PALETTE[0], lw=1.8, label="chaser")
    if plan:
        for m in plan.maneuvers:
            p = r[int(np.argmin(np.abs(rel.t - m.t)))]
            ax.scatter(*p, marker="^", s=40, color=PALETTE[0], edgecolor="white")
    ax.scatter(0, 0, 0, marker="s", s=50, color=TEXT, label="target")
    pts = [r, np.zeros((1, 3))]
    if keepout_km:
        X, Y, Z = _sphere(np.zeros(3), keepout_km)
        ax.plot_wireframe(X, Y, Z, color=TEXT_2, linewidth=0.3, alpha=0.5)
    _style_3d(ax, "km", title or "Relative motion (target LVLH)")
    ax.set_xlabel("V-bar [km]")
    ax.set_ylabel("-H-bar [km]")
    ax.set_zlabel("R-bar [km]")
    _set_equal_3d(ax, np.vstack(pts))
    ax.invert_zaxis()  # R-bar toward the primary drawn downward
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper left")
    return ax


def interactive_3d(trajs, filename: str, system: System = EARTH_MOON, units: str = "km",
                   origin: str = "barycenter", labels=None, title: str = "",
                   show_lpoints=("L1", "L2"), color_by: str | None = None,
                   speed_ref: str | None = None, speed_range=None, speed_scale: str = "linear"):
    """Write an interactive plotly HTML file (rotate/zoom/hover). Requires plotly.

    color_by='speed' colours each path by speed with a shared colour bar (see
    plot_trajectories_2d); hover then also shows the speed.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # ADR-3
        raise ImportError("interactive_3d requires plotly: pip install plotly") from exc
    if not isinstance(trajs, (list, tuple)):
        trajs = [trajs]
    k = _scale(system, units)
    rotating = all(getattr(t, "frame", "rotating") == "rotating" for t in trajs)
    o = _origin(system, origin) if rotating else np.zeros(3)
    fig = go.Figure()
    sp = _speed_setup(trajs, system, origin, color_by, speed_ref, speed_range, speed_scale)
    for n, tr in enumerate(trajs):
        r = (tr.r - o) * k
        days = tr.t_days
        if sp is not None:
            v = sp["values"][n]
            # plotly has no log colour scale: colour by log10(speed), label ticks in km/s
            cv = np.log10(np.maximum(v, 1e-3)) if sp["log"] else v
            lo, hi = sp["norm"].vmin, sp["norm"].vmax
            cmin, cmax = (np.log10(lo), np.log10(hi)) if sp["log"] else (lo, hi)
            cbar = dict(title=sp["label"], len=0.7)
            if sp["log"]:
                ticks = _speed_ticks(lo, hi)
                cbar.update(tickvals=np.log10(ticks).tolist(), ticktext=[f"{x:g}" for x in ticks])
            fig.add_trace(go.Scatter3d(
                x=r[:, 0], y=r[:, 1], z=r[:, 2], mode="lines",
                line=dict(color=cv, colorscale=[[q / (len(SPEED_RAMP) - 1), c] for q, c in enumerate(SPEED_RAMP)],
                          cmin=cmin, cmax=cmax, width=6, showscale=(n == 0), colorbar=cbar),
                name=labels[n] if labels else (tr.label or f"trajectory {n + 1}"),
                customdata=np.column_stack((days, v)),
                hovertemplate="x %{x:.0f}<br>y %{y:.0f}<br>z %{z:.0f}<br>t %{customdata[0]:.3f} d"
                              "<br>speed %{customdata[1]:.3f} km/s<extra>%{fullData.name}</extra>"))
            continue
        fig.add_trace(go.Scatter3d(
            x=r[:, 0], y=r[:, 1], z=r[:, 2], mode="lines",
            line=dict(color=series_color(n), width=4),
            name=labels[n] if labels else (tr.label or f"trajectory {n + 1}"),
            customdata=days,
            hovertemplate="x %{x:.0f}<br>y %{y:.0f}<br>z %{z:.0f}<br>t %{customdata:.3f} d"
                          "<extra>%{fullData.name}</extra>"))
    bodies = []
    if rotating:
        bodies = [(system.pos1, system.r1, EARTH_COLOR, system.name1),
                  (system.pos2, system.r2, MOON_COLOR, system.name2)]
    else:
        center = trajs[0].frame.split("-", 1)[-1]
        bodies = [(np.zeros(3), system.r1 if center == "earth" else system.r2,
                   EARTH_COLOR if center == "earth" else MOON_COLOR, center)]
    for pos, rad, col, name in bodies:
        X, Y, Z = _sphere((pos - o) * k if rotating else pos, rad / system.lstar * k, 40)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0, col], [1, col]],
                                 showscale=False, name=name, hoverinfo="name"))
    if rotating and show_lpoints:
        lp = cr3bp.lagrange_points(system.mu)
        names = [nm for nm in lp if nm in show_lpoints]
        q = np.array([(lp[nm] - o) * k for nm in names])
        fig.add_trace(go.Scatter3d(x=q[:, 0], y=q[:, 1], z=q[:, 2], mode="markers+text",
                                   text=names, marker=dict(size=3, color=TEXT_2),
                                   textfont=dict(color=TEXT_2), name="Lagrange points"))
    u = "km" if units == "km" else "nd"
    fig.update_layout(
        title=dict(text=title, x=0.02), paper_bgcolor="white",
        font=dict(color=TEXT),
        scene=dict(aspectmode="data",
                   xaxis=dict(title=f"x [{u}]", gridcolor=GRID, backgroundcolor="white"),
                   yaxis=dict(title=f"y [{u}]", gridcolor=GRID, backgroundcolor="white"),
                   zaxis=dict(title=f"z [{u}]", gridcolor=GRID, backgroundcolor="white")),
        legend=dict(x=0.01, y=0.98))
    fig.write_html(filename, include_plotlyjs="cdn")
    return fig
