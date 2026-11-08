"""Load docs/case_study_tess.html in Edge, check every figure renders, report errors.

    python docs/tools/check_case_study_tess.py docs/case_study_tess.html <outdir>
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="ascii", errors="replace")
url = Path(sys.argv[1]).resolve().as_uri()
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
errs = []
FIGS = ["fig-timeline", "fig-truthres", "fig-srp", "fig-pred", "fig-coast", "fig-rot", "fig-long", "fig-angle"]
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
            pg.locator(f"#{f}").locator("xpath=ancestor::div[contains(@class,'figure')]").screenshot(path=str(out / f"{theme}_{f}.png"))
        if theme == "light":
            for sec in ("summary", "night", "arc", "pred", "stable", "compare"):
                pg.locator(f"#{sec}").screenshot(path=str(out / f"sec_{sec}.png"))
            print("headline:", pg.inner_text("#headline"))
            for i in ("truthText", "srpText", "predText", "coastText", "longText", "compareTable"):
                print(f"{i}:", pg.inner_text(f"#{i}"))
        pg.close()
    b.close()
print("errors:", errs if errs else "none")
