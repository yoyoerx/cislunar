"""Satellite-servicing architecture analysis: base stations, routes and campaigns.

Building blocks
---------------
* ``Node``: an orbit that can host a station or a client. Either a periodic CR3BP
  orbit (free parameter: phase) or a Keplerian orbit about the Earth or Moon (free
  parameters: argument of latitude, RAAN, and inclination when a range is allowed).
* ``node_transfer(A, B)``: minimum two-impulse Delta-v from orbit A to a rendezvous
  in orbit B, optimised over where on each orbit the transfer starts/ends and over
  the time of flight ("phase-free": a servicer that can wait for good phasing).
  Lambert scan -> CR3BP shooting -> Nelder-Mead (ADR-20).
* ``earth_transfer``: two-body Hohmann + optimally split plane change between Earth
  orbits (accurate where the Moon's pull is negligible).
* ``coorbital_phasing``: Delta-v to reach a client elsewhere on the station's own orbit.
* ``cost_matrix``: all of the above for a set of nodes, in parallel.
* ``campaign``: Delta-v ledger and rocket-equation propellant / IMLEO for N servicing
  missions flown directly from LEO or from a base station (ADR-23).

Reference planes (ADR-21): CR3BP z is the Moon's orbit normal; Earth orbits are given
by equatorial inclination, with the equator inclined EPS_DEG to the Moon's orbit.
"""
from __future__ import annotations

import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import qmc

from . import cr3bp, frames, periodic as P, twobody
from .constants import EARTH_MOON, R_EARTH, R_MOON, System
from .targeting import lambert_candidates, shoot

EPS_DEG = 23.4  # Earth's equator relative to the Moon's orbit plane (nominal)
G0 = 9.80665e-3  # km/s^2
TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

@dataclass
class Node:
    key: str
    label: str
    category: str  # "earth" | "translunar" | "cislunar" | "lunar"
    orbit: P.PeriodicOrbit | None = None  # periodic CR3BP node
    center: str = "earth"  # Keplerian node: central body
    a_km: float = 0.0
    e: float = 0.0
    inc_deg: float = 0.0  # inclination to the Earth-Moon plane (Keplerian)
    inc_range: tuple | None = None  # if set, inclination is a free parameter in this range
    inc_eq_deg: float | None = None  # inclination to Earth's equator (Earth orbits)
    system: System = field(default=EARTH_MOON, repr=False)

    @property
    def periodic(self) -> bool:
        return self.orbit is not None

    @property
    def earth_kepler(self) -> bool:
        return not self.periodic and self.center == "earth"

    def bounds(self):
        if self.periodic:
            return [(0.0, 1.0)]
        b = [(0.0, TWO_PI), (0.0, TWO_PI)]
        if self.inc_range:
            b.append(tuple(np.radians(self.inc_range)))
        return b

    @property
    def ndim(self) -> int:
        return len(self.bounds())

    def state(self, p, t: float = 0.0) -> np.ndarray:
        """Rotating-frame state (nondim) at time t for node parameters p."""
        s = self.system
        if self.periodic:
            return self.orbit.state_at((p[0] % 1.0) * self.orbit.period)
        inc = p[2] if self.inc_range else np.radians(self.inc_deg)
        gm = s.gm1 if self.center == "earth" else s.gm2
        r, v = twobody.coe2rv(self.a_km, self.e, inc, p[1] % TWO_PI, 0.0, p[0] % TWO_PI, gm)
        x_in = np.concatenate((s.nd_length(r), s.nd_velocity(v)))
        return frames.inertial_to_rotating(t, x_in, s, self.center)

    @property
    def period_days(self) -> float:
        if self.periodic:
            return self.orbit.period_days
        gm = self.system.gm1 if self.center == "earth" else self.system.gm2
        return float(TWO_PI * np.sqrt(self.a_km ** 3 / gm) / 86400.0)

    def trajectory_km(self, n: int = 400, origin: str = "moon"):
        """Sampled orbit (km, rotating frame, relative to origin) for plotting."""
        s = self.system
        o = frames.center_position(s, origin)
        if self.periodic:
            return s.km(self.orbit.trajectory(n).r - o)
        inc = np.radians(np.mean(self.inc_range) if self.inc_range else self.inc_deg)
        p = [0.0, 0.0, inc] if self.inc_range else [0.0, 0.0]
        pts = []
        for u in np.linspace(0.0, TWO_PI, n):
            p[0] = u
            pts.append(self.state(p, 0.0)[:3] - o)
        return s.km(np.array(pts))


def default_nodes(system: System = EARTH_MOON) -> dict:
    """The station and client orbits used in the case study (cached per process)."""
    key = id(system)
    if key in _NODE_CACHE:
        return _NODE_CACHE[key]
    eps = EPS_DEG
    nodes = [
        Node("leo", "LEO depot (400 km, 28.5 deg)", "earth", a_km=R_EARTH + 400.0, inc_deg=28.5,
             inc_range=(abs(28.5 - eps), 28.5 + eps), inc_eq_deg=28.5, system=system),
        Node("sso", "LEO sun-synchronous (700 km, 98 deg)", "earth", a_km=R_EARTH + 700.0,
             inc_deg=98.0, inc_eq_deg=98.0, system=system),
        Node("meo", "MEO / GNSS (20,200 km, 55 deg)", "earth", a_km=R_EARTH + 20200.0,
             inc_deg=55.0, inc_eq_deg=55.0, system=system),
        Node("geo", "GEO (35,786 km, equatorial)", "earth", a_km=42164.0, inc_deg=eps,
             inc_eq_deg=0.0, system=system),
        Node("heo21", "Translunar 2:1 resonant (TESS-like)", "translunar",
             orbit=P.resonant(2, 1, 0.55, system), system=system),
        Node("nrho", "9:2 NRHO (Gateway)", "cislunar", orbit=P.nrho(system=system), system=system),
        Node("dro", "DRO (70,000 km)", "cislunar", orbit=P.dro(70000.0, system), system=system),
        Node("l1h", "L1 halo (15,000 km, north)", "cislunar",
             orbit=P.halo(1, 15000.0, "north", system), system=system),
        Node("l2h", "L2 halo (20,000 km, south)", "cislunar",
             orbit=P.halo(2, 20000.0, "south", system), system=system),
        Node("llo", "Low lunar orbit (100 km, polar)", "lunar", center="moon", a_km=R_MOON + 100.0,
             inc_deg=90.0, system=system),
    ]
    out = {n.key: n for n in nodes}
    _NODE_CACHE[key] = out
    return out


_NODE_CACHE: dict = {}

STATION_KEYS = ["leo", "geo", "heo21", "nrho", "dro", "l1h", "l2h", "llo"]
CLIENT_KEYS = ["sso", "meo", "geo", "heo21", "nrho", "dro", "l1h", "l2h", "llo"]


# ---------------------------------------------------------------------------
# Transfers
# ---------------------------------------------------------------------------

@dataclass
class TransferResult:
    origin: str
    dest: str
    dv1_mps: float
    dv2_mps: float
    tof_days: float
    method: str  # "cr3bp" | "two-body" | "coorbital"
    p_origin: list = field(default_factory=list)
    p_dest: list = field(default_factory=list)
    x_post: list = field(default_factory=list)  # state after departure burn (nondim)
    tof: float = 0.0  # nondim
    note: str = ""

    @property
    def total_mps(self) -> float:
        return self.dv1_mps + self.dv2_mps

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_mps"] = self.total_mps
        return d


def default_tof_days(A: Node, B: Node) -> tuple:
    cats = {A.category, B.category}
    if "earth" in cats:
        return (2.0, 7.0)
    if "lunar" in cats:
        return (1.5, 8.0)
    if "translunar" in cats:
        return (1.5, 14.0)
    return (1.0, 16.0)


def node_transfer(A: Node, B: Node, tof_days=None, n_scan: int = 4096, n_seeds: int = 12,
                  n_refine: int = 3, maxiter: int | None = None, seed: int = 0,
                  verbose: bool = False) -> TransferResult:
    """Minimum-Delta-v phase-free rendezvous transfer from orbit A to orbit B (CR3BP).

    1. Quasi-random (Sobol) scan over A's and B's parameters and the TOF, costed
       with two-body Lambert solutions about Earth and Moon (milliseconds each).
    2. The n_seeds best distinct seeds are converged by CR3BP shooting.
    3. The n_refine best are refined by Nelder-Mead over all parameters with
       warm-started shooting. Returns the best TransferResult.

    When A is a Keplerian orbit, each seed is also tried with tangential departure
    burns (the natural escape geometry from a low orbit), which converge far better
    than two-body Lambert guesses when the far end is perturbed. So put the low
    orbit at the departure end; the reverse leg follows by symmetry (ADR-22).
    """
    s = A.system
    if maxiter is None:
        maxiter = 160 if (not A.periodic or not B.periodic) else 400
    tb = tuple(s.nd_time_days(np.array(tof_days or default_tof_days(A, B))))
    bounds = A.bounds() + B.bounds() + [tb]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    nA = A.ndim

    def unpack(q):
        x = lo + np.clip(q, 0.0, 1.0) * (hi - lo)
        return x[:nA], x[nA:-1], x[-1]

    # 1. Lambert scan
    samp = qmc.Sobol(len(bounds), scramble=True, seed=seed).random(n_scan)
    scan = []
    for q in samp:
        pA, pB, tof = unpack(q)
        xA, xB = A.state(pA, 0.0), B.state(pB, tof)
        best = None
        for name, v1, v2 in lambert_candidates(xA, xB[:3], tof, s, 0.0):
            c = np.linalg.norm(v1 - xA[3:]) + np.linalg.norm(xB[3:] - v2)
            if best is None or c < best[0]:
                best = (c, v1)
        if best is not None and np.isfinite(best[0]):
            scan.append((best[0], q, best[1]))
    scan.sort(key=lambda r: r[0])

    # 2. converge distinct seeds in the CR3BP
    def evaluate(q, v_guess):
        pA, pB, tof = unpack(q)
        xA, xB = A.state(pA, 0.0), B.state(pB, tof)
        v, xf = shoot(xA, xB[:3], tof, s, v_guess=v_guess, max_iter=20)
        return (np.linalg.norm(v - xA[3:]), np.linalg.norm(xB[3:] - xf[3:]), v, xA, tof)

    seeds, chosen = [], []
    for c, q, v1 in scan:
        if len(chosen) >= n_seeds:
            break
        if any(np.linalg.norm(q - qq) < 0.08 for qq in chosen):
            continue
        chosen.append(q)
        guesses = [v1]
        if not A.periodic and len(chosen) <= 4:  # escape-burn seeds for the best few
            xA0 = A.state(unpack(q)[0], 0.0)
            gm = 1.0 - s.mu if A.center == "earth" else s.mu
            c0 = frames.center_position(s, A.center)
            rr = xA0[:3] - c0
            v_in = xA0[3:] + np.cross([0.0, 0.0, 1.0], rr)  # inertial, about the centre
            v_esc = np.sqrt(2 * gm / np.linalg.norm(rr))
            for f in (0.97, 1.0, 1.03):  # just below / at / above escape speed, prograde
                v_new = v_in / np.linalg.norm(v_in) * v_esc * f
                guesses.append(v_new - np.cross([0.0, 0.0, 1.0], rr))
        for vg in guesses:
            try:
                d1, d2, v, xA, tof = evaluate(q, vg)
            except (RuntimeError, np.linalg.LinAlgError):
                continue
            seeds.append((d1 + d2, q, v))
            if verbose:
                print(f"   seed: Lambert {s.mps(c):8.1f} -> CR3BP {s.mps(d1 + d2):8.1f} m/s")
    if not seeds:
        raise RuntimeError(f"no CR3BP transfer found {A.key} -> {B.key}")
    seeds.sort(key=lambda r: r[0])

    # 3. Nelder-Mead refinement
    best = {"cost": np.inf}
    for cost0, q0, v0 in seeds[:n_refine]:
        state = {"v": v0}

        def f(q):
            if np.any(q < -0.02) or np.any(q > 1.02):
                return 1e3
            try:
                d1, d2, v, xA, tof = evaluate(q, state["v"])
            except (RuntimeError, np.linalg.LinAlgError):
                return 1e3
            state["v"] = v
            c = d1 + d2
            if c < best["cost"]:
                best.update(cost=c, q=q.copy(), v=v.copy(), d1=d1, d2=d2, xA=xA, tof=tof)
            return c

        f(q0)
        simplex = [q0] + [q0 + 0.03 * np.eye(len(q0))[k] for k in range(len(q0))]
        minimize(f, q0, method="Nelder-Mead",
                 options={"maxfev": maxiter, "xatol": 1e-4, "fatol": 1e-6,
                          "initial_simplex": np.array(simplex)})
    pA, pB, tof = unpack(best["q"])
    x_post = best["xA"].copy()
    x_post[3:] = best["v"]
    return TransferResult(A.key, B.key, float(s.mps(best["d1"])), float(s.mps(best["d2"])),
                          float(s.days(tof)), "cr3bp", pA.tolist(), pB.tolist(),
                          x_post.tolist(), float(tof))


def hohmann_plane(r1_km: float, r2_km: float, di_rad: float, gm: float = 398600.435436):
    """Two-impulse Hohmann transfer between circular orbits with the plane change
    split optimally between the two burns. Returns (dv1, dv2) in km/s."""
    v1 = np.sqrt(gm / r1_km)
    v2 = np.sqrt(gm / r2_km)
    if abs(r1_km - r2_km) < 1e-6:
        return 2 * v1 * np.sin(di_rad / 2), 0.0
    a = 0.5 * (r1_km + r2_km)
    vp = np.sqrt(gm * (2 / r1_km - 1 / a))
    va = np.sqrt(gm * (2 / r2_km - 1 / a))

    def dvs(f):
        d1 = np.sqrt(v1 ** 2 + vp ** 2 - 2 * v1 * vp * np.cos(f * di_rad))
        d2 = np.sqrt(va ** 2 + v2 ** 2 - 2 * va * v2 * np.cos((1 - f) * di_rad))
        return d1, d2

    r = minimize_scalar(lambda f: sum(dvs(f)), bounds=(0.0, 1.0), method="bounded")
    return dvs(r.x)


def earth_transfer(A: Node, B: Node) -> TransferResult:
    """Earth orbit to Earth orbit: Hohmann with optimal plane split (two-body)."""
    di = np.radians(abs(A.inc_eq_deg - B.inc_eq_deg))
    d1, d2 = hohmann_plane(A.a_km, B.a_km, di, A.system.gm1)
    tof = np.pi * np.sqrt((0.5 * (A.a_km + B.a_km)) ** 3 / A.system.gm1) / 86400.0
    return TransferResult(A.key, B.key, float(d1 * 1000), float(d2 * 1000), float(tof), "two-body",
                          note=f"plane change {np.degrees(di):.1f} deg, RAANs aligned")


def coorbital_phasing(node: Node, dphase: float = 0.25, max_days: float = 30.0) -> TransferResult:
    """Delta-v to reach a client dphase of an orbit ahead on the same orbit, within
    max_days. Keplerian circular orbits: analytic phasing orbit; periodic CR3BP
    orbits: two-impulse CR3BP rendezvous searched over the flight time."""
    s = node.system
    if not node.periodic:
        gm = s.gm1 if node.center == "earth" else s.gm2
        T = TWO_PI * np.sqrt(node.a_km ** 3 / gm)
        k = max(1, int(max_days * 86400.0 / T))
        Tph = T * (1.0 - dphase / k)  # faster, lower phasing orbit catches up
        aph = (gm * (Tph / TWO_PI) ** 2) ** (1.0 / 3.0)
        vc = np.sqrt(gm / node.a_km)
        vph = np.sqrt(gm * (2 / node.a_km - 1 / aph))
        dv = abs(vc - vph) * 1000
        return TransferResult(node.key, node.key, float(dv), float(dv), float(k * Tph / 86400), "coorbital",
                              note=f"{k} phasing revolutions")
    # Periodic orbits: step onto a neighbouring member of the same family whose
    # period differs by dT = dphase * T / n (n revolutions within max_days), drift,
    # and step back. Each step costs the velocity difference between the two orbits
    # at their common symmetric x-z crossing (the small position offset is left to
    # station-keeping). Resonant (multi-crossing) orbits use the Keplerian estimate
    # dv = mu dT / (3 T a v_p) at perigee.
    orb = node.orbit
    T = orb.period
    n = max(1, int(s.nd_time_days(max_days) // T))
    dT = dphase * T / n
    x0 = orb.state0
    planar = abs(x0[P.IZ]) < 1e-12
    if node.category == "translunar":
        tr = orb.trajectory(4000)
        de = np.linalg.norm(tr.r - s.pos1, axis=1)
        k = int(np.argmin(de))
        rp = de[k]
        a = ((1.0 - s.mu) * (T / TWO_PI) ** 2) ** (1.0 / 3.0)
        vp = np.sqrt((1.0 - s.mu) * (2.0 / rp - 1.0 / a))
        dv = (1.0 - s.mu) * dT / (3.0 * T * a * vp)
    else:
        free = (P.IX, P.IVY) if planar else (P.IX, P.IZ, P.IVY)
        targets = (P.IVX,) if planar else (P.IVX, P.IVZ)
        dv = None
        while dv is None:  # longer drift (more revs, smaller period change) until a neighbour exists
            for sign in (-1.0, 1.0):
                try:
                    nb = P.correct_symmetric(x0, s, free=free, targets=targets, period=T + sign * dT)
                except RuntimeError:
                    continue
                dv = np.linalg.norm(nb.state0[3:] - x0[3:])
                dT = sign * dT
                break
            if dv is None:
                n *= 2
                dT = dphase * T / n
                if n > 64:
                    raise RuntimeError(f"no neighbouring family member for phasing on {node.key}")
    # Cross-check with the drift-rate estimate dv_total = (2/3) ds / t (exact for circular
    # Keplerian phasing): ds = dphase x path length, t = drift time. Where a family's period
    # hardly changes with size (small halos) the neighbour step is unrealistically large
    # and the drift estimate is used instead (flagged in the note).
    t_drift = n * abs(T - dT)
    path = orb.trajectory(2000).r
    length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    dv_drift = (2.0 / 3.0) * dphase * length / t_drift / 2.0  # per burn
    note = f"neighbouring family member, period {s.days(T - dT):.2f} d, {n} revs"
    if dv > 5.0 * dv_drift:
        note = (f"drift estimate (2/3)ds/t over {s.days(t_drift):.0f} d; family-neighbour step "
                f"would cost {2 * s.mps(dv):.0f} m/s (period nearly constant along the family)")
        dv = dv_drift
    dv_mps = float(s.mps(dv))
    return TransferResult(node.key, node.key, dv_mps, dv_mps, float(s.days(t_drift)),
                          "coorbital", [0.0], [dphase], note=note)


def _pair_task(args):
    a, b, kw = args
    nodes = default_nodes()
    A, B = nodes[a], nodes[b]
    try:
        if a == b:
            return coorbital_phasing(A).to_dict()
        if A.earth_kepler and B.earth_kepler:
            return earth_transfer(A, B).to_dict()
        return node_transfer(A, B, **kw).to_dict()
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        r = TransferResult(a, b, float("nan"), float("nan"), float("nan"), "failed", note=str(exc))
        return r.to_dict()


def cost_matrix(keys=None, workers: int | None = None, symmetric: bool = True,
                cache: str | None = None, verbose: bool = False, **kw) -> dict:
    """Transfer costs between all pairs of nodes (and co-orbital phasing on the
    diagonal). With symmetric=True only one direction of each pair is computed and
    mirrored (ADR-22). Returns {"a->b": TransferResult dict}.

    Runs in parallel processes (call from under `if __name__ == "__main__":` on
    Windows). With cache=path, finished pairs are saved as they complete and reused on
    the next call. Pairs whose direct search fails are filled with the cheapest two-leg
    route through another node (an upper bound, flagged in the note).
    """
    nodes = default_nodes()
    keys = list(keys or nodes.keys())
    # Earth orbits first (lowest first), then lunar orbits, then periodic orbits, so the
    # low orbit is the departure end of each pair (escape-burn seeds; ADR-20)
    keys.sort(key=lambda k: (nodes[k].periodic, nodes[k].center != "earth",
                             nodes[k].a_km if not nodes[k].periodic else 0.0))
    done = {}
    if cache and os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            done = json.load(fh)
    tasks = []
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if symmetric and j < i:
                continue
            if f"{a}->{b}" in done and done[f"{a}->{b}"]["method"] != "failed":
                continue
            tasks.append((a, b, kw))

    def save():
        if cache:
            with open(cache, "w", encoding="utf-8") as fh:
                json.dump(done, fh)

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1 or len(tasks) <= 1:
        for t in tasks:
            done[f"{t[0]}->{t[1]}"] = _pair_task(t)
            save()
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_pair_task, t): t for t in tasks}
            for k, fut in enumerate(as_completed(futs)):
                a, b, _ = futs[fut]
                done[f"{a}->{b}"] = fut.result()
                save()
                if verbose:
                    r = done[f"{a}->{b}"]
                    print(f"  [{k + 1}/{len(tasks)}] {a}->{b}: {r['total_mps']:.0f} m/s ({r['method']})",
                          flush=True)
    out = {}
    for k, r in done.items():
        a, b = k.split("->")
        if a not in keys or b not in keys:
            continue
        out[k] = r
        if symmetric and a != b:
            rr = dict(r)
            rr.update(origin=b, dest=a, dv1_mps=r["dv2_mps"], dv2_mps=r["dv1_mps"],
                      note=(r.get("note", "") + " (mirror of reverse leg)").strip(),
                      x_post=[], p_origin=r["p_dest"], p_dest=r["p_origin"])
            out[f"{b}->{a}"] = rr
    # fill failed pairs with the best two-leg route through another node
    for k, r in list(out.items()):
        if r["method"] != "failed":
            continue
        a, b = k.split("->")
        best = None
        for m in keys:
            if m in (a, b):
                continue
            r1, r2 = out.get(f"{a}->{m}"), out.get(f"{m}->{b}")
            if not r1 or not r2 or "failed" in (r1["method"], r2["method"])                     or r1["method"] == "two-leg" or r2["method"] == "two-leg":
                continue
            tot = r1["total_mps"] + r2["total_mps"]
            if best is None or tot < best[0]:
                best = (tot, m, r1, r2)
        if best:
            tot, m, r1, r2 = best
            out[k] = dict(r, method="two-leg", dv1_mps=r1["total_mps"], dv2_mps=r2["total_mps"], total_mps=tot,
                          tof_days=r1["tof_days"] + r2["tof_days"],
                          note=f"direct search failed; upper bound via {m} ({r1['total_mps']:.0f} + "
                               f"{r2['total_mps']:.0f} m/s)")
    return out


# ---------------------------------------------------------------------------
# Campaign economics
# ---------------------------------------------------------------------------

# Station upkeep (station-keeping), m/s per year. Cislunar / translunar values are the
# p95 of the toolkit's Floquet-mode Monte Carlo (lower bounds: no solar gravity); LEO,
# GEO and LLO are typical literature values, NOT computed by the toolkit.
STATION_UPKEEP_MPS_YR = {
    "nrho": 1.4, "l1h": 8.3, "l2h": 5.8, "dro": 0.0, "heo21": 0.0,
    "leo": 50.0,  # drag make-up at 400 km (literature, solar-cycle dependent)
    "geo": 50.0,  # north-south station-keeping (literature)
    "llo": 100.0,  # lunar mascons at 100 km (literature)
}
LUNAR_ASCENT_MPS = 1870.0  # lunar surface -> LLO, typical published value (not computed)


@dataclass
class MassModel:
    station_kg: float = 20000.0  # station / depot dry mass delivered once
    servicer_kg: float = 2000.0  # reusable servicer (dry, incl. robotics)
    kit_kg: float = 500.0  # servicing payload left at each client (propellant, parts, pod)
    isp_s: float = 320.0  # all in-space propulsion (storable bipropellant)
    servicer_life: int = 20  # missions per servicer before replacement
    allowance_frac: float = 0.03  # per-leg phasing / navigation / finite-burn allowance
    allowance_mps: float = 10.0
    resupply_batch: int = 5  # missions supplied per logistics flight (Delta-v ledger only)
    station_years: float = 10.0  # station life, for upkeep propellant
    lunar_propellant: bool = False  # servicer + upkeep propellant mined on the Moon

    @property
    def c(self) -> float:  # exhaust velocity, m/s
        return self.isp_s * G0 * 1000.0

    def leg(self, dv_mps: float) -> float:
        return dv_mps * (1 + self.allowance_frac) + self.allowance_mps

    def gear(self, dv_mps: float) -> float:
        """Initial / final mass for a burn sequence of total dv (rocket equation)."""
        return float(np.exp(dv_mps / self.c))


LAUNCH_SITE_INC_DEG = 28.5  # lowest inclination the launcher reaches directly (Cape Canaveral)


def direct_dv_mps(client: str, costs: dict) -> float:
    """Delta-v of the direct architecture's in-space leg for a client.

    Earth-orbit clients: the servicer is launched straight into a 400 km parking orbit
    in the client's own plane (or the launch-site minimum, 28.5 deg, for lower
    inclinations such as GEO) and flies a Hohmann transfer with the remaining plane
    change. Other clients: the LEO depot orbit -> client route from the matrix.
    A precomputed value under costs["direct->client"] takes precedence.
    """
    if f"direct->{client}" in costs:
        return costs[f"direct->{client}"]["total_mps"]
    node = default_nodes().get(client)
    if node is not None and node.earth_kepler:
        inc_park = max(node.inc_eq_deg, LAUNCH_SITE_INC_DEG)
        d1, d2 = hohmann_plane(R_EARTH + 400.0, node.a_km, np.radians(inc_park - node.inc_eq_deg),
                               node.system.gm1)
        return float((d1 + d2) * 1000.0)
    return costs[f"leo->{client}"]["total_mps"]


def _dv(costs: dict, a: str, b: str) -> float:
    return 0.0 if a == b and a == "leo_free" else costs[f"{a}->{b}"]["total_mps"]


def campaign(n: int, client: str, station: str | None, costs: dict,
             mm: MassModel | None = None) -> dict:
    """Totals for n servicing missions to one client class.

    station=None: direct architecture. Each mission launches a servicer + kit to a
    400 km parking orbit (in the client's plane for Earth-orbit clients) and flies to
    the client one way; the servicer stays with the client (see direct_dv_mps).
    station=key: station fielded once (LEO -> station), a reusable servicer flies
    station -> client -> station each mission, kits and servicer propellant are
    resupplied from LEO, servicers replaced every servicer_life missions.
    Returns Delta-v ledger (m/s, summed over all flights) and mass totals (kg).
    """
    mm = mm or MassModel()
    ms, mk = mm.servicer_kg, mm.kit_kg
    if station is None:
        dv = mm.leg(direct_dv_mps(client, costs))
        imleo = n * (ms + mk) * mm.gear(dv)
        return {"architecture": "direct", "station": None, "client": client, "n": n,
                "dv_per_mission": dv, "ledger": {"servicing": n * dv, "fielding": 0.0, "logistics": 0.0},
                "dv_ledger": n * dv, "imleo_kg": imleo, "propellant_kg": imleo - n * (ms + mk),
                "imleo_per_mission": imleo / n}
    d_ls = 0.0 if station == "leo" else mm.leg(_dv(costs, "leo", station))
    d_out = mm.leg(_dv(costs, station, client))
    d_back = mm.leg(_dv(costs, client, station))
    prop_mission = (ms * mm.gear(d_back) + mk) * mm.gear(d_out) - ms - mk
    n_serv = int(np.ceil(n / mm.servicer_life))
    sk = STATION_UPKEEP_MPS_YR.get(station, 0.0) * mm.station_years
    sk_prop = mm.station_kg * (mm.gear(sk) - 1.0)
    imleo_field = (mm.station_kg + ms) * mm.gear(d_ls)
    imleo_serv = (n_serv - 1) * ms * mm.gear(d_ls)
    lunar_kg = 0.0
    if mm.lunar_propellant:  # Earth ships only kits; propellant comes up from the Moon
        d_ms = LUNAR_ASCENT_MPS + (0.0 if station == "llo" else mm.leg(_dv(costs, "llo", station)))
        lunar_kg = (n * prop_mission + sk_prop) * mm.gear(d_ms)
        imleo_ops = n * mk * mm.gear(d_ls)
        imleo_sk = 0.0
    else:
        imleo_ops = n * (prop_mission + mk) * mm.gear(d_ls)
        imleo_sk = sk_prop * mm.gear(d_ls)
    n_flights = int(np.ceil(n / mm.resupply_batch))
    ledger = {"fielding": d_ls, "servicing": n * (d_out + d_back),
              "logistics": (n_flights + n_serv - 1) * d_ls, "upkeep": sk}
    imleo = imleo_field + imleo_serv + imleo_ops + imleo_sk
    return {"architecture": "station", "station": station, "client": client, "n": n,
            "dv_per_mission": d_out + d_back, "dv_field": d_ls, "ledger": ledger,
            "dv_ledger": sum(ledger.values()), "imleo_kg": imleo,
            "imleo_field_kg": imleo_field, "imleo_ops_kg": imleo_ops + imleo_serv,
            "imleo_upkeep_kg": imleo_sk, "lunar_propellant_kg": lunar_kg,
            "propellant_kg": imleo - (mm.station_kg + n_serv * ms + n * mk),
            "imleo_per_mission": imleo / n, "prop_per_mission_kg": prop_mission}


def break_even(client: str, station: str, costs: dict, mm: MassModel | None = None,
               n_max: int = 100000):
    """Smallest number of missions at which the station architecture needs less mass
    in LEO than flying every mission directly (None if never within n_max)."""
    mm = mm or MassModel()
    lo, hi = 1, n_max
    f = lambda n: campaign(n, client, station, costs, mm)["imleo_kg"] - campaign(n, client, None, costs, mm)["imleo_kg"]
    if f(hi) > 0:
        return None
    if f(lo) <= 0:
        return 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return hi
