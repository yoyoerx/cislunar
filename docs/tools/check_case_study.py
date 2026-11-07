"""Load docs/case_study_base_stations.html in Edge, exercise controls, report errors.

    python docs/tools/check_case_study.py docs/case_study_base_stations.html <outdir>
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="ascii", errors="replace")
url = Path(sys.argv[1]).resolve().as_uri()
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
errs = []
FIGS = ["fig-map", "fig-matrix", "fig-route", "fig-field", "fig-camp", "fig-camppm", "fig-ledger", "fig-mix"]
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge")
    for theme in ("light", "dark"):
        pg = b.new_page(viewport={"width": 1400, "height": 900}, color_scheme=theme)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto(url)
        pg.wait_for_timeout(3000)
        pg.add_style_tag(content="header.top{position:static !important}")
        for f in FIGS:
            n = pg.evaluate(f"document.getElementById('{f}').data ? document.getElementById('{f}').data.length : -1")
            if n < 1:
                errs.append(f"{theme}: {f} empty")
        for f in FIGS:
            pg.locator(f"#{f}").locator("xpath=ancestor::div[contains(@class,'figure')]").screenshot(path=str(out / f"{theme}_{f}.png"))
        if theme == "light":
            pg.locator("#summary").screenshot(path=str(out / "summary.png"))
            pg.locator("#best").screenshot(path=str(out / "best.png"))
            pg.locator("#breakeven").screenshot(path=str(out / "breakeven.png"))
            print("headline:", pg.inner_text("#headline")[:900])
            for c in ("sso", "geo", "nrho", "llo"):
                for n in ("10", "100", "1000"):
                    pg.select_option("#cClient", c)
                    pg.select_option("#cN", n)
                    pg.wait_for_timeout(200)
                    print(f"  {c} N={n}:", pg.inner_text("#campTiles").replace("\n", " | ")[:200])
            pg.check("#mLunar")
            pg.wait_for_timeout(300)
            print("  lunar prop (llo,1000):", pg.inner_text("#campTiles").replace("\n", " | ")[:220])
            pg.uncheck("#mLunar")
            print("mix:", pg.inner_text("#mixTiles").replace("\n", " | "))
            pg.evaluate("RT.sel = 'llo->l1h'; routeDraw();")
            pg.wait_for_timeout(500)
            print("route:", pg.inner_text("#routeTiles").replace("\n", " | "))
        pg.close()
    b.close()
print("errors:", errs if errs else "none")
