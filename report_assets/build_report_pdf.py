from pathlib import Path
import re

import markdown


ROOT = Path(__file__).resolve().parents[1]
MAIN_REPORT = ROOT / "RESEARCH_REPORT.md"
APPENDIX = ROOT / "report_assets" / "experiment_metrics_appendix.md"
HTML_OUTPUT = ROOT / "RESEARCH_REPORT_PRINT.html"


def build_html() -> None:
    main = MAIN_REPORT.read_text(encoding="utf-8")
    appendix = APPENDIX.read_text(encoding="utf-8")
    main = re.sub(r"~~~mermaid.*?~~~", "", main, flags=re.DOTALL)
    appendix = re.sub(
        r"^# Experiment Metrics Appendix",
        "## Appendix A — Complete experiment metrics ledger",
        appendix,
        count=1,
        flags=re.MULTILINE,
    )
    source = main + "\n\n" + appendix
    body = markdown.markdown(source, extensions=["tables", "fenced_code", "toc"])
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Kidney Stone Detection Research Report</title>
<style>
@page {{ size: A4; margin: 15mm 14mm 16mm 14mm; }}
:root {{ --ink:#2d3142; --muted:#4f5d75; --rule:#c8ccd4; --accent:#eb6c36; }}
* {{ box-sizing:border-box; }}
html,body {{ margin:0; padding:0; background:white; color:var(--ink); font-family:Arial, Helvetica, sans-serif; line-height:1.42; font-size:10.5pt; }}
body {{ max-width:190mm; margin:0 auto; }}
h1 {{ font-size:25pt; line-height:1.05; margin:0 0 12pt; font-weight:700; }}
h2 {{ font-size:16pt; line-height:1.15; border-bottom:1px solid var(--rule); padding-bottom:4pt; margin:20pt 0 8pt; page-break-after:avoid; }}
h2#appendix-a-complete-experiment-metrics-ledger {{ page-break-before:always; }}
h3 {{ font-size:12pt; margin:14pt 0 5pt; page-break-after:avoid; }}
p {{ margin:5pt 0 8pt; }}
ul,ol {{ margin-top:4pt; }}
blockquote {{ border-left:3px solid var(--accent); margin:10pt 0; padding:6pt 12pt; background:#fff7f2; color:var(--ink); }}
code,pre {{ font-family:"Cascadia Mono", Consolas, monospace; font-size:8pt; }}
pre {{ background:#f2f3f5; border:1px solid #d9dce2; padding:8pt; overflow-wrap:anywhere; white-space:pre-wrap; }}
code {{ background:#f2f3f5; padding:1pt 3pt; }}
table {{ width:100%; border-collapse:collapse; margin:7pt 0 12pt; font-size:8.4pt; page-break-inside:auto; }}
thead {{ display:table-header-group; }}
tr {{ page-break-inside:avoid; }}
th {{ text-align:left; background:#eef0f3; color:var(--ink); font-weight:700; }}
th,td {{ border:1px solid #d4d8df; padding:4pt 5pt; vertical-align:top; }}
img {{ display:block; max-width:100%; height:auto; margin:8pt auto 3pt; }}
p:has(img) {{ page-break-inside:avoid; }}
em {{ color:var(--muted); }}
a {{ color:#2e5aa8; text-decoration:none; }}
hr {{ border:0; border-top:1px solid var(--rule); margin:14pt 0; }}
@media print {{ a {{ color:inherit; }} img {{ break-inside:avoid; }} }}
</style>
</head>
<body>
{body}
</body>
</html>
"""
    HTML_OUTPUT.write_text(html, encoding="utf-8")
    print(f"Wrote {HTML_OUTPUT} ({len(html)} characters)")


if __name__ == "__main__":
    build_html()

