"""Load docs/case_study_2010kq.html in Edge, check every figure renders, report errors and texts.

    python docs/tools/check_case_study_kq.py docs/case_study_2010kq.html <outdir>
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="ascii", errors="replace")
url = Path(sys.argv[1]).resolve().as_uri()
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
errs = []
FIGS = ["fig-timeline", "fig-profile", "fig-valid", "fig-enc", "fig-tiss", "fig-frac", "fig-examples", "fig-sus", "fig-ae", "fig-future"]
TEXTS = ["headline", "dataText", "fitText", "fitTable", "validText", "encText", "luna1Text", "cloudText1", "cloudTiles", "cloudText2",
         "susText1", "susRefTable", "susText2", "verdictTable", "verdictText", "futureText", "limitsList"]
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
            for sec in ("summary", "mystery", "cloud", "suspects", "verdict"):
                pg.locator(f"#{sec}").screenshot(path=str(out / f"sec_{sec}.png"))
            for i in TEXTS:
                print(f"--- {i}:", pg.inner_text(f"#{i}"))
        pg.close()
    b.close()
print("errors:", errs if errs else "none")
