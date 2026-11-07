import sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

page_url = Path(sys.argv[1]).resolve().as_uri()
out = Path(sys.argv[2])
theme = sys.argv[3] if len(sys.argv) > 3 else "light"
errors = []
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge")
    pg = b.new_page(viewport={"width": 1400, "height": 900}, color_scheme=theme)
    pg.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    t = time.time()
    pg.goto(page_url, wait_until="load")
    pg.wait_for_timeout(2500)
    pg.add_style_tag(content="header.top{position:static !important}")
    print(f"loaded in {time.time() - t:.1f}s")
    for fid in ["fig-kepler", "fig-frame-rot", "fig-zvc", "fig-family", "fig-stab", "fig-sandbox",
                "fig-porkchop", "fig-pc3d", "fig-leo", "fig-cw", "fig-rpod", "fig-range"]:
        el = pg.locator(f"#{fid}")
        n = pg.evaluate(f"document.getElementById('{fid}').data ? document.getElementById('{fid}').data.length : -1")
        print(f"{fid}: traces={n}")
    for fid in ["fig-pc3d", "fig-zvc", "fig-family", "fig-sandbox", "fig-porkchop", "fig-leo", "fig-cw", "fig-rpod"]:
        pg.locator(f"#{fid}").locator("xpath=..").screenshot(path=str(out / f"{theme}_{fid}.png"))
    pg.locator("#frame").screenshot(path=str(out / f"{theme}_frames.png"))
    pg.screenshot(path=str(out / f"{theme}_top.png"))
    b.close()
print("ERRORS:" if errors else "no console errors")
for e in errors:
    print(" ", e[:300])
