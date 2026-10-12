"""Render the project status board from status/data/*.csv into easy-to-read tables.

Sources (edit these; one row per task):
  status/data/completed.csv    status/data/in_progress.csv    status/data/pending.csv
Outputs:
  status/COMPLETED.html  status/IN_PROGRESS.html  status/PENDING.html   (one styled table each, open in a browser)
  status/COMPARISON.html  (before vs now + tip methods vs your 40 human frames)
  status/STATUS_BOARD.html                                              (all three on one page)
  status/STATUS_BOARD.xlsx                                              (Excel: one sheet per state)

  python scripts/update_status_board.py
"""

import html
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ST = ROOT / "status"
SECTIONS = [("completed", "Completed", "#1e7d32", "COMPLETED.html"),
            ("in_progress", "In progress", "#e08a00", "IN_PROGRESS.html"),
            ("pending", "Pending", "#1565c0", "PENDING.html"),
            ("comparison", "Before vs now", "#6a1b9a", "COMPARISON.html"),
            ("tip_methods", "Tip methods vs your 40 frames", "#00838f", "COMPARISON.html")]
CSS = """
body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f6f7f9;color:#1f2328}
h1{font-size:22px;margin:0 0 4px} .sub{color:#57606a;margin-bottom:18px}
h2{font-size:18px;margin:28px 0 8px;padding:6px 12px;border-radius:6px;color:#fff;display:inline-block}
table{border-collapse:collapse;width:100%;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08);font-size:14px}
th{background:#24292f;color:#fff;text-align:left;padding:9px 10px;position:sticky;top:0}
td{padding:8px 10px;border-top:1px solid #e4e7eb;vertical-align:top}
tr:nth-child(even) td{background:#fafbfc} tr:hover td{background:#eef4ff}
.pill{padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600;color:#fff;white-space:nowrap}
.High{background:#c62828}.Medium{background:#ef6c00}.Low{background:#607d8b}.Blocked{background:#6d4c41}
td.best{font-weight:700;color:#1e7d32}
.nav a{margin-right:14px}
"""


def table_html(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = "" if pd.isna(r[c]) else str(r[c])
            cls = ' class="best"' if v in ("CURRENT BEST", "TRAINING NOW") else ""
            v = f'<span class="pill {html.escape(v)}">{html.escape(v)}</span>' if c == "Priority" else html.escape(v)
            cells.append(f"<td{cls}>{v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def page(title: str, body: str, stamp: str) -> str:
    links = [("STATUS_BOARD.html", "All"), ("COMPLETED.html", "Completed"), ("IN_PROGRESS.html", "In progress"),
             ("PENDING.html", "Pending"), ("COMPARISON.html", "Comparison")]
    nav = " | ".join(f'<a href="{f}">{t}</a>' for f, t in links)
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta http-equiv='refresh' content='60'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
            f"<h1>PRISM-Endo — {html.escape(title)}</h1><div class='sub'>Last updated {stamp} · "
            f"main goal: DeepLabCut tip model · page refreshes every 60 s</div><div class='nav'>{nav}</div>{body}"
            f"</body></html>")


def main():
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    data = {k: pd.read_csv(ST / "data" / f"{k}.csv", dtype=str, keep_default_na=False) for k, *_ in SECTIONS}
    allbody, files = [], {}
    for key, title, colour, fname in SECTIONS:
        df = data[key]
        section = f"<h2 style='background:{colour}'>{title} ({len(df)})</h2>{table_html(df)}"
        files.setdefault(fname, []).append((title, section))
        allbody.append(section)
    for fname, parts in files.items():
        title = "Comparison" if len(parts) > 1 else parts[0][0]
        (ST / fname).write_text(page(title, "".join(p for _, p in parts), stamp), encoding="utf-8")
    (ST / "STATUS_BOARD.html").write_text(page("Status board", "".join(allbody), stamp), encoding="utf-8")

    from openpyxl.styles import Alignment, Font, PatternFill

    with pd.ExcelWriter(ST / "STATUS_BOARD.xlsx", engine="openpyxl") as xw:
        for key, title, colour, _ in SECTIONS:
            df = data[key]
            df.to_excel(xw, sheet_name=title, index=False)
            ws = xw.sheets[title]
            fill = PatternFill("solid", fgColor=colour.lstrip("#"))
            for cell in ws[1]:
                cell.font, cell.fill = Font(bold=True, color="FFFFFF"), fill
            for i, col in enumerate(df.columns, start=1):
                width = min(70, max(10, int(max([len(col)] + [len(str(v)) for v in df[col]]) * 1.05)))
                ws.column_dimensions[ws.cell(1, i).column_letter].width = width
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
            ws.freeze_panes = "A2"
            ws.cell(len(df) + 3, 1, f"Last updated {stamp}")
    print(f"status board updated {stamp}: " + ", ".join(f"{t} {len(data[k])}" for k, t, *_ in SECTIONS))


if __name__ == "__main__":
    main()
