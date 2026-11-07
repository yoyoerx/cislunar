import sys
sys.stdout.reconfigure(encoding='ascii', errors='replace')
from pathlib import Path
from playwright.sync_api import sync_playwright
url = Path(sys.argv[1]).resolve().as_uri(); out = Path(sys.argv[2])
errs = []
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge")
    for theme in ("light", "dark"):
        pg = b.new_page(viewport={"width": 1400, "height": 900}, color_scheme=theme)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto(url); pg.wait_for_timeout(2500)
        pg.add_style_tag(content="header.top{position:static !important}")
        if theme == "light":
            # sandbox presets
            for pre in ["l1kick", "lloesc", "leotli", "drokick"]:
                pg.click(f"[data-preset={pre}]"); pg.wait_for_timeout(2500)
                tiles = pg.inner_text("#sbTiles").replace("\n", " | ")
                print(pre, "->", tiles)
                pg.locator("#fig-sandbox").locator("xpath=..").screenshot(path=str(out / f"sb_{pre}.png"))
            # LEO TLI error
            pg.eval_on_selector("#leoErr", "e => { e.value = 5; e.dispatchEvent(new Event('input')); }"); pg.wait_for_timeout(2500)
            print("leo +5 m/s ->", pg.inner_text("#leoTiles").replace("\n", " | "))
            pg.locator("#fig-leo").locator("xpath=..").screenshot(path=str(out / "leo_err.png"))
            # CW presets
            for pre in ["pro", "football", "hop"]:
                pg.click(f"[data-cw={pre}]"); pg.wait_for_timeout(300)
                print("cw", pre, "->", pg.inner_text("#cwMsg"))
            pg.locator("#fig-cw").locator("xpath=..").screenshot(path=str(out / "cw_hop.png"))
            # RPOD aborts + time + zoom
            for k in range(2): pg.check(f"[data-ab='{k}']")
            pg.eval_on_selector("#rpT", "e => { e.value = 700; e.dispatchEvent(new Event('input')); }")
            pg.wait_for_timeout(500)
            pg.locator("#fig-rpod").locator("xpath=..").screenshot(path=str(out / "rpod.png"))
            pg.select_option("#rpZoom", "1200"); pg.wait_for_timeout(500)
            pg.locator("#fig-rpod").screenshot(path=str(out / "rpod_zoom.png"))
            # porkchop click: simulate selecting another cell via plotly event
            pg.evaluate("PC.sel = [5, 15]; pcDraw();"); pg.wait_for_timeout(1500)
            print("porkchop cell ->", pg.inner_text("#pcTiles").replace("\n", " | "))
            # frames play a bit
            pg.click("#frPlay"); pg.wait_for_timeout(2000); pg.click("#frPlay")
            pg.locator("#frame").locator(".figure").screenshot(path=str(out / "frames_play.png"))
        else:
            for fid in ["fig-kepler", "fig-family", "fig-porkchop", "fig-rpod"]:
                pg.locator(f"#{fid}").locator("xpath=..").screenshot(path=str(out / f"dark_{fid}.png"))
            pg.locator("#welcome").screenshot(path=str(out / "dark_welcome.png"))
        pg.close()
    b.close()
print("errors:", errs if errs else "none")
