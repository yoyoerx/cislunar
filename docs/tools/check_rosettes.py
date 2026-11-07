"""Exercise the rosette section of docs/manual.html in Edge and screenshot it.

    python docs/tools/check_rosettes.py docs/manual.html <outdir>
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="ascii", errors="replace")
url = Path(sys.argv[1]).resolve().as_uri()
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
errs = []
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge")
    for theme in ("light", "dark"):
        pg = b.new_page(viewport={"width": 1400, "height": 900}, color_scheme=theme)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto(url)
        pg.wait_for_timeout(2500)
        pg.add_style_tag(content="header.top{position:static !important}")
        sel = pg.locator("#rsSel option")
        print(theme, "resonances:", sel.count(), " lunar:", pg.locator("#lrSel option").count())
        if theme == "light":
            for k in range(sel.count()):
                pg.select_option("#rsSel", str(k))
                pg.wait_for_timeout(200)
                print("  ", pg.inner_text("#rsTiles").replace("\n", " | "))
            pg.select_option("#rsSel", "1")
            pg.check("#rsGuess")
            pg.wait_for_timeout(1500)
            print("  guess:", pg.inner_text("#rsTiles").split("|")[-1].replace("\n", " | "))
            pg.locator("#fig-ros-rot").locator("xpath=../..").screenshot(path=str(out / "ros_guess.png"))
            pg.uncheck("#rsGuess")
            pg.select_option("#rsSel", "3")
            pg.click("#rsPlay")
            pg.wait_for_timeout(2500)
            pg.click("#rsPlay")
            print("  mid-play:", pg.inner_text("#rsTo"))
            pg.locator("#fig-ros-rot").locator("xpath=../..").screenshot(path=str(out / "ros_play.png"))
            for key in ("nrho", "dro", "l2h"):
                pg.select_option("#lrSel", key)
                pg.wait_for_timeout(800)
                print("  lunar", key, pg.inner_text("#lrNo"))
                pg.locator("#fig-lr-in").locator("xpath=../..").screenshot(path=str(out / f"lunar_{key}.png"))
            pg.locator("#rosettes").screenshot(path=str(out / "ros_section.png"))
        else:
            pg.select_option("#rsSel", "0")
            pg.wait_for_timeout(300)
            pg.locator("#fig-ros-rot").locator("xpath=../..").screenshot(path=str(out / "ros_dark.png"))
        pg.close()
    b.close()
print("errors:", errs if errs else "none")
