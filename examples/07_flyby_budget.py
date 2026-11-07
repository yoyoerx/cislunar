"""Flybys and Delta-v budgets.

Scenario: an inspector parked in a 70,000 km DRO flies past an object in the
9:2 NRHO at a 25 km miss distance and returns to the DRO.

1. Flyby targeting at several miss distances (departure Delta-v vs relative speed).
2. Return leg: from the post-flyby state back to a rendezvous with the DRO.
3. Statistical TCMs for both legs (linear covariance, checked by Monte Carlo).
4. Station-keeping: Floquet-mode Monte Carlo for the NRHO, an L2 halo and the DRO.
5. The budget roll-up.
"""
import numpy as np

import cislunar as cl
from cislunar import navigation as N, periodic as P, targeting as T
from cislunar.budget import DeltaVBudget

S = cl.EARTH_MOON
dro, nrho = cl.dro(70000.0), cl.nrho()

# 1. flyby at a miss distance -------------------------------------------------
td, tof = S.nd_time_days(7.33), S.nd_time_days(9.0)
print("1) Flyby of the NRHO object from the DRO")
for miss in (0.0, 25.0, 100.0):
    fb = T.flyby(dro, nrho, td, tof, miss_km=miss)
    print("  " + fb.summary().replace("\n", "\n  "))
fb = T.flyby(dro, nrho, td, tof, miss_km=25.0)

# 2. return leg ---------------------------------------------------------------
print("2) Return to the DRO after the flyby")
x_after = fb.transfer.x_arrive
t_ret = td + tof


def coast(t):  # the inspector's state after the flyby (no burn at the flyby)
    return cl.cr3bp.propagate_state(x_after, t - t_ret, S)


best = None
for wait_d in (0.0, 1.0, 2.0):
    for tof_r in np.linspace(3.0, 12.0, 10):
        try:
            tr = T.intercept(coast, dro, t_ret + S.nd_time_days(wait_d), S.nd_time_days(tof_r), S)
        except RuntimeError:
            continue
        if best is None or tr.total_mps < best.total_mps:
            best = tr
print("  " + best.summary().replace("\n", "\n  "))

# 3. statistical TCMs -----------------------------------------------------------
print("3) TCMs: departure-burn execution error 1% magnitude, 0.5 deg pointing")
legs = {"outbound": (fb.transfer, None), "return": (best, best.dv2)}
tcm_results = {}
for name, (tr, dva) in legs.items():
    x_post = tr.x_chaser.copy()
    x_post[3:] += tr.dv1
    P0 = N.burn_execution_covariance(tr.dv1, 0.01, 0.5, S)
    times = S.nd_time_days(np.array([0.25, 0.5, 1.0, 2.0, 4.0]))
    res = N.linear_tcm(x_post, tr.tof, P0, times, S, dv_arrival=dva)
    tcm_results[name] = res
    print(f"  {name} leg (departure {tr.dv1_mps:.1f} m/s):")
    print("  " + res.table().replace("\n", "\n  "))
x_post = fb.transfer.x_chaser.copy()
x_post[3:] += fb.transfer.dv1
mc = N.monte_carlo_tcm(x_post, fb.transfer.tof,
                       N.burn_execution_covariance(fb.transfer.dv1, 0.01, 0.5, S),
                       S.nd_time_days(np.array([0.5, 5.0])), S, n=200)
print("  outbound Monte Carlo, TCMs at day 0.5 and 5.0:")
print("  " + mc.summary().replace("\n", "\n  "))

# 4. station-keeping ------------------------------------------------------------
print("4) Station-keeping (nav 1 km / 1 cm/s, execution 2%)")
sk = {}
for name, orb, iv in (("9:2 NRHO", nrho, 6.56), ("L2 halo 20,000 km", P.halo(2, 20000.0, "south"), 2.0),
                      ("DRO 70,000 km", dro, 7.0)):
    orb.family = name
    sk[name] = N.stationkeeping_mc(orb, iv, duration_days=180, n=16)
    print("  " + sk[name].summary())

# 5. budget ---------------------------------------------------------------------
b = DeltaVBudget("DRO inspector: NRHO flyby at 25 km and return", margin_frac=0.10)
b.add_transfer(fb.transfer, "outbound flyby", flyby=True)
b.add_transfer(best, "return to DRO")
b.add_tcm(tcm_results["outbound"], "outbound TCM", index=1)
b.add_tcm(tcm_results["return"], "return TCM", index=1)
b.add_stationkeeping(sk["DRO 70,000 km"], years=2.0, name="DRO station-keeping")
print()
print(b.table())
