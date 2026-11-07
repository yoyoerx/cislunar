"""Reference-frame transformations.

* Rotating (synodic, barycentric) <-> inertial, centred on the barycentre or
  either primary. The inertial frame coincides with the rotating frame at
  t = 0 (plus an optional phase theta0).
* LVLH frame of a target relative to a primary (Orion / NASA convention):
    x = V-bar = j x k, y = -H-bar (negative orbit normal), z = R-bar (to primary)
  built from the target's primary-relative state in the rotating frame.
"""
from __future__ import annotations

import numpy as np

from . import cr3bp
from .constants import EARTH_MOON, System

_ZHAT = np.array([0.0, 0.0, 1.0])


def center_position(system: System, center: str) -> np.ndarray:
    center = center.lower()
    if center in ("barycenter", "barycentre", "bary"):
        return np.zeros(3)
    if center in ("earth", "primary1", system.name1.lower()):
        return system.pos1
    if center in ("moon", "primary2", system.name2.lower()):
        return system.pos2
    raise ValueError(f"unknown center {center!r}")


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotating_to_inertial(t, x, system: System = EARTH_MOON, center: str = "earth",
                         theta0: float = 0.0):
    """Rotating-frame state(s) -> inertial state(s) about `center`.

    t: scalar or (N,); x: (6,) or (N, 6). Nondimensional in and out.
    """
    t = np.atleast_1d(np.asarray(t, float))
    X = np.atleast_2d(np.asarray(x, float))
    c = center_position(system, center)
    out = np.empty_like(X)
    for k in range(len(X)):
        R = rot_z(t[k] + theta0)
        rr = X[k, :3] - c
        out[k, :3] = R @ rr
        out[k, 3:] = R @ (X[k, 3:] + np.cross(_ZHAT, rr))
    return out[0] if np.ndim(x) == 1 else out


def inertial_to_rotating(t, x_in, system: System = EARTH_MOON, center: str = "earth",
                         theta0: float = 0.0):
    """Inertial state(s) about `center` -> rotating-frame state(s)."""
    t = np.atleast_1d(np.asarray(t, float))
    X = np.atleast_2d(np.asarray(x_in, float))
    c = center_position(system, center)
    out = np.empty_like(X)
    for k in range(len(X)):
        Rt = rot_z(t[k] + theta0).T
        rr = Rt @ X[k, :3]
        out[k, :3] = rr + c
        out[k, 3:] = Rt @ X[k, 3:] - np.cross(_ZHAT, rr)
    return out[0] if np.ndim(x_in) == 1 else out


def trajectory_to_inertial(traj: cr3bp.Trajectory, center: str = "earth",
                           theta0: float = 0.0) -> cr3bp.Trajectory:
    xi = rotating_to_inertial(traj.t, traj.x, traj.system, center, theta0)
    return cr3bp.Trajectory(t=traj.t, x=xi, system=traj.system,
                            label=traj.label, frame=f"inertial-{center}")


def primary_track_inertial(t, system: System = EARTH_MOON, body: str = "moon",
                           center: str = "earth", theta0: float = 0.0):
    """Inertial positions (N, 3) of a primary as seen from `center`."""
    t = np.atleast_1d(t)
    pos = center_position(system, body)
    x = np.tile(np.concatenate((pos, np.zeros(3))), (len(t), 1))
    return rotating_to_inertial(t, x, system, center, theta0)[:, :3]


# ---------------------------------------------------------------------------
# LVLH
# ---------------------------------------------------------------------------

def _unit_and_rate(w, wdot):
    n = np.linalg.norm(w)
    u = w / n
    return u, (wdot - u * np.dot(u, wdot)) / n


def lvlh_matrix(x_target, system: System = EARTH_MOON, primary: str = "moon",
                with_rate: bool = False):
    """Rotation C (rows = LVLH axes in rotating-frame components), optionally
    with its time derivative Cdot (exact, using the CR3BP acceleration)."""
    x_target = np.asarray(x_target, float)
    r = x_target[:3] - center_position(system, primary)
    v = x_target[3:6]
    k, kd = _unit_and_rate(-r, -v)
    h = np.cross(r, v)
    if not with_rate:
        j = -h / np.linalg.norm(h)
        i = np.cross(j, k)
        return np.vstack((i, j, k))
    a = cr3bp.acceleration(x_target, system.mu)
    j, jd = _unit_and_rate(-h, -np.cross(r, a))
    i = np.cross(j, k)
    idot = np.cross(jd, k) + np.cross(j, kd)
    return np.vstack((i, j, k)), np.vstack((idot, jd, kd))


def to_lvlh(x_target, x_chaser, system: System = EARTH_MOON, primary: str = "moon"):
    """Chaser state relative to target, in target LVLH. Returns (6,) nondim."""
    C, Cd = lvlh_matrix(x_target, system, primary, with_rate=True)
    dr = np.asarray(x_chaser[:3]) - x_target[:3]
    dv = np.asarray(x_chaser[3:6]) - x_target[3:6]
    return np.concatenate((C @ dr, C @ dv + Cd @ dr))


def from_lvlh(x_target, rel_lvlh, system: System = EARTH_MOON, primary: str = "moon"):
    """Inverse of to_lvlh: LVLH relative state -> chaser rotating-frame state."""
    C, Cd = lvlh_matrix(x_target, system, primary, with_rate=True)
    rho = np.asarray(rel_lvlh[:3], float)
    rhod = np.asarray(rel_lvlh[3:6], float)
    dr = C.T @ rho
    dv = C.T @ (rhod - Cd @ dr)
    return np.concatenate((x_target[:3] + dr, x_target[3:6] + dv))


def lvlh_vector_to_rotating(x_target, vec, system: System = EARTH_MOON,
                            primary: str = "moon"):
    """Rotate a free vector (e.g. a delta-v) from LVLH to rotating axes."""
    return lvlh_matrix(x_target, system, primary).T @ np.asarray(vec, float)


def rotating_vector_to_lvlh(x_target, vec, system: System = EARTH_MOON,
                            primary: str = "moon"):
    return lvlh_matrix(x_target, system, primary) @ np.asarray(vec, float)


def speed_kms(x, system: System = EARTH_MOON, reference: str = "moon"):
    """Speed (km/s) of rotating-frame state(s) x (6,) or (N, 6).

    reference = "moon" / "earth": speed relative to that body in a non-rotating
    frame (what KSP calls orbital speed); "rotating": speed in the rotating frame
    (the speed that enters the Jacobi constant).
    """
    X = np.atleast_2d(np.asarray(x, float))
    if reference == "rotating":
        v = X[:, 3:6]
    else:
        rr = X[:, :3] - center_position(system, reference)
        v = X[:, 3:6] + np.cross(_ZHAT, rr)
    out = system.kms(np.linalg.norm(v, axis=1))
    return out[0] if np.ndim(x) == 1 else out
