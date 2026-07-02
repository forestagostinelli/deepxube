from pathlib import Path
from html import escape
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
CONF_PATH = ROOT / "docs_gen" / "conf.py"
OUT_DIR = ROOT / "docs"


spec = importlib.util.spec_from_file_location("docs_conf", CONF_PATH)
conf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conf)

latest = getattr(conf, "smv_latest_version", "main")
redirect_target = getattr(conf, "smv_rename_latest_version", latest)

target = f"{redirect_target}/index.html"

html = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url={escape(target)}">
    <link rel="canonical" href="{escape(target)}">
    <title>Redirecting...</title>
  </head>
  <body>
    <p>Redirecting to <a href="{escape(target)}">the latest documentation</a>.</p>
  </body>
</html>
"""

OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "index.html").write_text(html, encoding="utf-8")
print(f"Wrote docs/index.html -> {target}")