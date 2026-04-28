"""
SAIT Attendance Gap Detector
Scenario A — AI Automation Engineer Evaluation
Author: Candidate
"""

import os
import io
import json
import math
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import requests as http_requests
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER, TA_LEFT

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COLUMN_ALIASES = {
    "name":    ["name", "student name", "student", "full name", "studentname"],
    "roll":    ["roll", "roll no", "roll number", "id", "student id", "enrollment", "reg no", "regno"],
    "present": ["present", "days present", "attended", "attendance", "present days", "days attended"],
    "total":   ["total", "total days", "working days", "total working days", "days total", "total classes"],
}


def detect_column(df_columns, aliases):
    lower_cols = {c.lower().strip(): c for c in df_columns}
    for alias in aliases:
        if alias in lower_cols:
            return lower_cols[alias]
        for col_lower, col_orig in lower_cols.items():
            if alias in col_lower:
                return col_orig
    return None


def parse_dataframe(df):
    """Detect columns and return normalised list of student dicts."""
    name_col    = detect_column(df.columns, COLUMN_ALIASES["name"])
    roll_col    = detect_column(df.columns, COLUMN_ALIASES["roll"])
    present_col = detect_column(df.columns, COLUMN_ALIASES["present"])
    total_col   = detect_column(df.columns, COLUMN_ALIASES["total"])

    if not name_col or not present_col:
        raise ValueError(
            f"Could not detect required columns (Name, Present). "
            f"Found: {list(df.columns)}"
        )

    students = []
    for i, row in df.iterrows():
        name    = str(row[name_col]).strip() if name_col else f"Student {i+1}"
        roll    = str(row[roll_col]).strip()  if roll_col else str(i + 1)
        present = float(row[present_col]) if pd.notna(row[present_col]) else 0
        total   = float(row[total_col])   if (total_col and pd.notna(row[total_col])) else None
        students.append({"name": name, "roll": roll, "present": present, "total": total})

    return students


def compute_stats(students, threshold, total_override=None):
    """Add pct, deficit, flagged to each student record."""
    # If no total_override and no total in data, infer from max present value
    if not total_override:
        all_totals = [s["total"] for s in students if s["total"]]
        all_presents = [s["present"] for s in students if s["present"]]
        inferred_total = max(all_totals) if all_totals else (max(all_presents) * 1.2 if all_presents else 100)
    else:
        inferred_total = None

    results = []
    for s in students:
        if total_override:
            total = total_override
        elif s["total"]:
            total = s["total"]
        else:
            total = inferred_total or 100

        total = float(total)
        pct     = round((s["present"] / total) * 100, 1) if total else 0
        needed  = math.ceil((threshold / 100) * total)
        deficit = max(0, needed - int(s["present"]))
        results.append({
            **s,
            "total":   total,
            "pct":     pct,
            "deficit": deficit,
            "flagged": pct < threshold,
        })
    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Accept file + params, return JSON student list."""
    threshold     = float(request.form.get("threshold", 75))
    total_override = request.form.get("total_days", "").strip()
    total_override = float(total_override) if total_override else None

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded."}), 400

    filename = file.filename.lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Only CSV and Excel files are supported."}), 400

        students = parse_dataframe(df)
        results  = compute_stats(students, threshold, total_override)
        return jsonify({"students": results, "threshold": threshold})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/warning", methods=["POST"])
def generate_warning():
    """Generate an AI warning message for a single flagged student."""
    data      = request.json or {}
    name      = data.get("name", "Student")
    roll      = data.get("roll", "N/A")
    present   = data.get("present", 0)
    total     = data.get("total", 100)
    pct       = data.get("pct", 0)
    deficit   = data.get("deficit", 0)
    threshold = data.get("threshold", 75)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not set. Add it to your .env file."}), 400

    prompt = (
        f"Write a concise, formal academic warning letter body for a student "
        f"with low attendance. Keep it under 90 words. Do not include a date, "
        f"greeting header, subject line, or signature block — only the paragraph body.\n\n"
        f"Student name: {name}\n"
        f"Roll number: {roll}\n"
        f"Attendance: {present} out of {total} days ({pct}%)\n"
        f"Required threshold: {threshold}%\n"
        f"Days short: {deficit}\n\n"
        f"Tone: firm but supportive. Mention the consequence of not meeting the threshold "
        f"and encourage the student to improve."
    )

    try:
        resp = http_requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-opus-4-5",
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"]
        return jsonify({"warning": text})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/export/csv", methods=["POST"])
def export_csv():
    """Return a downloadable CSV report."""
    data      = request.json or {}
    students  = data.get("students", [])
    threshold = data.get("threshold", 75)

    rows = []
    for s in students:
        rows.append({
            "Name":           s["name"],
            "Roll No":        s["roll"],
            "Days Present":   int(s["present"]),
            "Total Days":     int(s["total"]),
            "Attendance (%)": s["pct"],
            "Deficit Days":   s["deficit"],
            "Status":         "FLAGGED" if s["flagged"] else "Clear",
            "Warning Message": s.get("warning", ""),
        })

    df  = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype="text/csv",
                     as_attachment=True,
                     download_name="attendance_report.csv")


@app.route("/export/pdf", methods=["POST"])
def export_pdf():
    """Return a downloadable PDF report."""
    data      = request.json or {}
    students  = data.get("students", [])
    threshold = data.get("threshold", 75)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=2*cm,    bottomMargin=2*cm)

    styles  = getSampleStyleSheet()
    title_s = ParagraphStyle("title", parent=styles["Heading1"],
                             alignment=TA_CENTER, fontSize=16, spaceAfter=6)
    sub_s   = ParagraphStyle("sub",   parent=styles["Normal"],
                             alignment=TA_CENTER, fontSize=10,
                             textColor=colors.grey, spaceAfter=16)
    warn_s  = ParagraphStyle("warn",  parent=styles["Normal"],
                             fontSize=8, leading=11)

    story = [
        Paragraph("Student Attendance Report", title_s),
        Paragraph(f"Threshold: {threshold}%  |  "
                  f"Flagged: {sum(1 for s in students if s['flagged'])} / {len(students)} students",
                  sub_s),
    ]

    # Table header
    header = ["Name", "Roll No", "Present", "Total", "%", "Deficit", "Status"]
    rows   = [header]
    for s in students:
        rows.append([
            s["name"],
            s["roll"],
            str(int(s["present"])),
            str(int(s["total"])),
            f"{s['pct']}%",
            f"-{s['deficit']} days" if s["deficit"] else "—",
            "FLAGGED" if s["flagged"] else "Clear",
        ])

    col_widths = [5.5*cm, 2.5*cm, 1.8*cm, 1.8*cm, 1.8*cm, 2.2*cm, 2.4*cm]
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        # Header
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#1a3c5e")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 9),
        ("ALIGN",        (0, 0), (-1, 0), "CENTER"),
        ("BOTTOMPADDING",(0, 0), (-1, 0), 8),
        ("TOPPADDING",   (0, 0), (-1, 0), 8),
        # Body
        ("FONTSIZE",     (0, 1), (-1, -1), 8),
        ("ALIGN",        (2, 1), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f5f7fa")]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#dde1e7")),
        ("TOPPADDING",   (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 1), (-1, -1), 5),
    ]))

    # Highlight flagged rows red
    for i, s in enumerate(students, start=1):
        if s["flagged"]:
            tbl.setStyle(TableStyle([
                ("TEXTCOLOR", (6, i), (6, i), colors.HexColor("#c0392b")),
                ("FONTNAME",  (6, i), (6, i), "Helvetica-Bold"),
                ("BACKGROUND",(0, i), (-1, i), colors.HexColor("#fff5f5")),
            ]))

    story.append(tbl)

    # Warning messages section
    flagged_with_warnings = [s for s in students if s.get("flagged") and s.get("warning")]
    if flagged_with_warnings:
        story.append(Spacer(1, 0.8*cm))
        story.append(Paragraph("AI-Generated Warning Messages", styles["Heading2"]))
        story.append(Spacer(1, 0.3*cm))
        for s in flagged_with_warnings:
            story.append(Paragraph(
                f"<b>{s['name']} ({s['roll']})</b> — {s['pct']}% attendance",
                styles["Normal"]
            ))
            story.append(Spacer(1, 0.15*cm))
            story.append(Paragraph(s["warning"].replace("\n", "<br/>"), warn_s))
            story.append(Spacer(1, 0.4*cm))

    doc.build(story)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf",
                     as_attachment=True,
                     download_name="attendance_report.pdf")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
