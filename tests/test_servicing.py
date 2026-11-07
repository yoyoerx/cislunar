import numpy as np
import pytest

from cislunar import cr3bp, frames, servicing as SV
from cislunar.constants import EARTH_MOON as S


@pytest.fixture(scope="module")
def nodes():
    return SV.default_nodes()


def test_nodes_and_states(nodes):
    assert set(SV.STATION_KEYS) <= set(nodes) and set(SV.CLIENT_KEYS) <= set(nodes)
    geo = nodes["geo"]
    x = geo.state([0.3, 1.1], t=0.7)
    r = np.linalg.norm(x[:3] - S.pos1)
    assert S.km(r) == pytest.approx(42164.0, rel=1e-9)
    # inertial speed about Earth equals the circular speed
    assert frames.speed_kms(x, S, "earth") == pytest.approx(np.sqrt(S.gm1 / 42164.0), rel=1e-9)
    # inclination to the Earth-Moon plane is the 23.4 deg equator tilt
    rr = x[:3] - S.pos1
    v_in = x[3:] + np.cross([0, 0, 1], rr)
    h = np.cross(rr, v_in)
    assert np.degrees(np.arccos(h[2] / np.linalg.norm(h))) == pytest.approx(SV.EPS_DEG, abs=1e-6)
    leo = nodes["leo"]
    assert leo.ndim == 3 and leo.bounds()[2][0] == pytest.approx(np.radians(28.5 - SV.EPS_DEG))
    assert nodes["nrho"].ndim == 1
    n = nodes["nrho"]
    assert np.allclose(n.state([0.0]), n.orbit.state0)


def test_hohmann_plane_textbook():
    d1, d2 = SV.hohmann_plane(6778.137, 42164.0, np.radians(28.5))
    assert d1 + d2 == pytest.approx(4.196, abs=0.01)  # LEO 28.5 deg -> GEO, km/s
    d1, d2 = SV.hohmann_plane(7000.0, 7000.0, np.radians(10.0))
    assert d1 == pytest.approx(2 * np.sqrt(398600.435436 / 7000.0) * np.sin(np.radians(5)))
    # optimal split beats doing the whole plane change at either end
    full_first = SV.hohmann_plane(6778.137, 42164.0, 0.0)
    assert d1 >= 0 and sum(full_first) < 4.196


def test_coorbital_phasing(nodes):
    geo = SV.coorbital_phasing(nodes["geo"])
    assert 5 < geo.total_mps < 40
    nrho = SV.coorbital_phasing(nodes["nrho"])
    assert 5 < nrho.total_mps < 100 and "neighbouring" in nrho.note
    l1h = SV.coorbital_phasing(nodes["l1h"])
    assert l1h.total_mps < 50 and "drift estimate" in l1h.note


def test_node_transfer_dro_nrho(nodes):
    r = SV.node_transfer(nodes["dro"], nodes["nrho"], n_scan=1024, n_seeds=6, n_refine=1, maxiter=120)
    assert 350 < r.total_mps < 800
    # the returned departure state really reaches the NRHO
    xf = cr3bp.propagate_state(np.array(r.x_post), r.tof, S)
    xt = nodes["nrho"].state(r.p_dest)
    assert np.linalg.norm(xf[:3] - xt[:3]) < 1e-8
    assert S.mps(np.linalg.norm(xt[3:] - xf[3:])) == pytest.approx(r.dv2_mps, rel=1e-6)


def test_campaign_economics():
    costs = {"direct->geo": {"total_mps": 4200.0}, "leo->nrho": {"total_mps": 3950.0},
             "nrho->geo": {"total_mps": 1700.0}, "geo->nrho": {"total_mps": 1700.0}}
    mm = SV.MassModel(allowance_frac=0.0, allowance_mps=0.0)
    d = SV.campaign(10, "geo", None, costs, mm)
    assert d["dv_ledger"] == pytest.approx(42000.0)
    assert d["imleo_kg"] == pytest.approx(10 * 2500 * np.exp(4200 / mm.c))
    st = SV.campaign(10, "geo", "nrho", costs, mm)
    assert st["dv_per_mission"] == pytest.approx(3400.0)
    # station costs scale ~linearly in n after fielding; break-even is consistent
    n_be = SV.break_even("geo", "nrho", costs, mm)
    if n_be is not None:
        assert SV.campaign(n_be, "geo", "nrho", costs, mm)["imleo_kg"] <= \
            SV.campaign(n_be, "geo", None, costs, mm)["imleo_kg"]
        if n_be > 1:
            assert SV.campaign(n_be - 1, "geo", "nrho", costs, mm)["imleo_kg"] > \
                SV.campaign(n_be - 1, "geo", None, costs, mm)["imleo_kg"]


def test_cost_matrix_cache_mirror_and_fallback(tmp_path):
    cache = tmp_path / "routes.json"
    keys = ["leo", "meo", "geo"]  # analytic two-body pairs: fast
    out = SV.cost_matrix(keys, workers=1, cache=str(cache))
    assert len(out) == 9 and cache.exists()
    assert out["geo->leo"]["total_mps"] == pytest.approx(out["leo->geo"]["total_mps"])
    assert out["geo->leo"]["dv1_mps"] == pytest.approx(out["leo->geo"]["dv2_mps"])
    # a failed direct pair is filled with the best two-leg route
    import json
    d = json.loads(cache.read_text(encoding="utf-8"))
    d["leo->geo"] = SV.TransferResult("leo", "geo", float("nan"), float("nan"), float("nan"),
                                      "failed", note="forced").to_dict()
    cache.write_text(json.dumps(d), encoding="utf-8")
    monkey = SV._pair_task
    try:
        SV._pair_task = lambda t: d[f"{t[0]}->{t[1]}"]  # keep it failed on recompute
        out2 = SV.cost_matrix(keys, workers=1, cache=str(cache))
    finally:
        SV._pair_task = monkey
    r = out2["leo->geo"]
    assert r["method"] == "two-leg" and "via meo" in r["note"]
    assert r["total_mps"] == pytest.approx(out["leo->meo"]["total_mps"] + out["meo->geo"]["total_mps"])


def test_direct_baseline_uses_client_plane():
    costs = {}
    sso = SV.direct_dv_mps("sso", costs)   # launched into SSO plane: just 400 -> 700 km
    assert 100 < sso < 250
    geo = SV.direct_dv_mps("geo", costs)   # launch site limits: 28.5 deg plane change remains
    assert geo == pytest.approx(4196, abs=15)
