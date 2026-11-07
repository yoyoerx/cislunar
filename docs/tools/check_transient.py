"""Exercise section 11 (transient orbits, flybys, budgets) of docs/manual.html in Edge.

    python docs/tools/check_transient.py docs/manual.html <outdir>
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="ascii", errors="replace")
url = Path(sys.argv[1]).resolve().as_uri()
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
errs = []


def shot(pg, fid, name):
    pg.locator(f"#{fid}").locator("xpath=ancestor::div[contains(@class,'figure')]").screenshot(path=str(out / name))


with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge")
    for theme in ("light", "dark"):
        pg = b.new_page(viewport={"width": 1400, "height": 900}, color_scheme=theme)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto(url)
        pg.wait_for_timeout(3000)
        pg.add_style_tag(content="header.top{position:static !important}")
        for fid in ("fig-efold", "fig-manifold", "fig-fatemap", "fig-fatepath", "fig-fbgrid", "fig-fbrange",
                    "fig-tcm", "fig-tcmmc", "fig-sk", "fig-skhist", "fig-budget"):
            n = pg.evaluate(f"document.getElementById('{fid}').data ? document.getElementById('{fid}').data.length : -1")
            if n < 1:
                errs.append(f"{theme}: {fid} has no traces")
        if theme == "dark":
            for fid in ("fig-manifold", "fig-fatemap", "fig-budget"):
                shot(pg, fid, f"dark_{fid}.png")
            pg.close()
            continue
        for fid in ("fig-efold", "fig-manifold", "fig-fatemap", "fig-fbgrid", "fig-tcm", "fig-sk", "fig-budget"):
            shot(pg, fid, f"{fid}.png")
        print("mf:", pg.inner_text("#mfTiles").replace("\n", " | "))
        pg.select_option("#mfOrbit", "l2halo")
        pg.select_option("#mfKind", "stable")
        pg.wait_for_timeout(800)
        print("mf halo stable:", pg.inner_text("#mfTiles").replace("\n", " | "))
        shot(pg, "fig-manifold", "manifold_halo.png")
        # click a fate-map cell via the plotly event path
        pg.evaluate("FM_SEL.i = 30; FM_SEL.j = 50; fmDraw();")
        pg.wait_for_timeout(1500)
        print("fm:", pg.inner_text("#fmTiles").replace("\n", " | "))
        pg.select_option("#fmKind", "ftle")
        pg.select_option("#fmSense", "prograde")
        pg.wait_for_timeout(1500)
        shot(pg, "fig-fatemap", "fatemap_ftle_pro.png")
        for m in ("vrel", "tot"):
            pg.select_option("#fbMetric", m)
            pg.wait_for_timeout(300)
        print("fb table rows:", pg.locator("#fbTable tr").count())
        for leg in ("outbound", "return", "leo"):
            pg.select_option("#tcLeg", leg)
            pg.wait_for_timeout(300)
            print("tcm", leg, ":", pg.inner_text("#tcTiles").replace("\n", " | "))
        pg.eval_on_selector("#tcScale", "e => { e.value = 2; e.dispatchEvent(new Event('input')); }")
        pg.wait_for_timeout(300)
        print("tcm leo x2:", pg.inner_text("#tcTiles").replace("\n", " | "))
        for k in range(pg.locator("#skHist option").count()):
            pg.select_option("#skHist", str(k))
            pg.wait_for_timeout(300)
        shot(pg, "fig-skhist", "sk_hist.png")
        print("budget A:", pg.inner_text("#bgTiles").replace("\n", " | "))
        pg.select_option("#bgScen", "B")
        pg.wait_for_timeout(500)
        print("budget B:", pg.inner_text("#bgTiles").replace("\n", " | "))
        pg.select_option("#bgMiss", "500")
        pg.eval_on_selector("#bgExe", "e => { e.value = 2; e.dispatchEvent(new Event('input')); }")
        pg.wait_for_timeout(300)
        print("budget B (500 km, x2 exec):", pg.inner_text("#bgTiles").replace("\n", " | "))
        shot(pg, "fig-budget", "budget_B.png")
        pg.close()
    b.close()
print("errors:", errs if errs else "none")
