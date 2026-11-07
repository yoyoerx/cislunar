"""Exercise the speed heat-map option in docs/manual.html (Figures 4, 5, 10, 11).

    python docs/tools/check_speed.py docs/manual.html <outdir>
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="ascii", errors="replace")
url = Path(sys.argv[1]).resolve().as_uri()
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
errs = []
FIGS = [("fam", "fig-family"), ("sb", "fig-sandbox"), ("rs", "fig-ros-rot"), ("lr", "fig-lr-rot")]
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge")
    for theme in ("light", "dark"):
        pg = b.new_page(viewport={"width": 1400, "height": 900}, color_scheme=theme)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto(url)
        pg.wait_for_timeout(3000)
        pg.add_style_tag(content="header.top{position:static !important}")
        for prefix, fid in FIGS:
            for ref in ("moon", "earth", "rot"):
                pg.select_option(f"#{prefix}Spd", ref)
                pg.wait_for_timeout(900 if prefix != "sb" else 2500)
            pg.select_option(f"#{prefix}Spd", "moon")
            pg.wait_for_timeout(900 if prefix != "sb" else 2500)
            has = pg.evaluate(f"""(() => {{ const d = document.getElementById('{fid}').data;
                return d.some(tr => (tr.line && Array.isArray(tr.line.color)) || (tr.marker && Array.isArray(tr.marker.color))); }})()""")
            if not has:
                errs.append(f"{theme}: {fid} has no speed-coloured trace")
            pg.locator(f"#{fid}").locator("xpath=ancestor::div[contains(@class,'figure')]").screenshot(path=str(out / f"{theme}_{prefix}.png"))
        if theme == "light":
            print("family tiles:", pg.inner_text("#famTiles").replace("\n", " | "))
            pg.uncheck("#famSpdLog")
            pg.wait_for_timeout(500)
            pg.select_option("#famSpd", "off")
            pg.wait_for_timeout(500)
            n = pg.evaluate("document.getElementById('fig-family').data.length")
            print("family plain traces:", n)
        pg.close()
    b.close()
print("errors:", errs if errs else "none")
