"""Cislunar orbit design, visualization and RPOD targeting toolkit.

Core dynamics are the Earth-Moon CR3BP in the nondimensional barycentric
rotating frame; see architecture.md.
"""
from . import (budget, cr3bp, frames, lambert, manifolds, navigation, od, periodic, relative,
               targeting, twobody)
from .constants import EARTH_MOON, System
from .cr3bp import Trajectory, jacobi, lagrange_points, propagate
from .periodic import PeriodicOrbit, dro, halo, lyapunov, nrho
from .relative import Waypoint, plan_waypoints
from .budget import DeltaVBudget
from .targeting import flyby, intercept, optimize_transfer, porkchop

__all__ = [
    "EARTH_MOON", "System", "Trajectory", "PeriodicOrbit", "Waypoint",
    "budget", "cr3bp", "frames", "lambert", "manifolds", "navigation", "od", "periodic",
    "relative", "targeting", "twobody", "DeltaVBudget", "flyby",
    "propagate", "jacobi", "lagrange_points", "lyapunov", "halo", "nrho", "dro",
    "intercept", "porkchop", "optimize_transfer", "plan_waypoints",
]
__version__ = "0.1.0"
