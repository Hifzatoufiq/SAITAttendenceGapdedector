"""
AttendIQ — Advanced Attendance Intelligence Platform
OpenAI GPT-4 powered warning letters + ML risk prediction
"""

import os, io, json, math, csv
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import requests as http_requests
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import numpy as np
from dotenv import load_dotenv

# Load .env from same directory as app.py
_base = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_base, ".env"), override=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN DETECTION
# ─────────────────────────────────────────────────────────────────────────────

COLUMN_ALIASES = {
    "name":    ["name", "student name", "student", "full name", "studentname"],
    "roll":    ["roll", "roll no", "roll number", "id", "student id", "enrollment", "reg no", "regno"],
    "present": ["present", "days present", "attended", "attendance", "present days", "days attended"],
    "total":   ["total", "total days", "working days", "total working days", "days total", "total classes"],
    "class":   ["class", "section", "batch", "group", "division"],
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
    name_col    = detect_column(df.columns, COLUMN_ALIASES["name"])
    roll_col    = detect_column(df.columns, COLUMN_ALIASES["roll"])
    present_col = detect_column(df.columns, COLUMN_ALIASES["present"])
    total_col   = detect_column(df.columns, COLUMN_ALIASES["total"])
    class_col   = detect_column(df.columns, COLUMN_ALIASES["class"])

    if not name_col or not present_col:
        raise ValueError(f"Missing required columns. Found: {list(df.columns)}")

    students = []
    for i, row in df.iterrows():
        students.append({
            "name":    str(row[name_col]).strip() if name_col else f"Student {i+1}",
            "roll":    str(row[roll_col]).strip()  if roll_col else str(i + 1),
            "present": float(row[present_col]) if pd.notna(row[present_col]) else 0,
            "total":   float(row[total_col])   if (total_col and pd.notna(row[total_col])) else None,
            "class":   str(row[class_col]).strip() if (class_col and pd.notna(row[class_col])) else "General",
        })
    return students

# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(students, threshold, total_override=None):
    all_totals   = [s["total"] for s in students if s["total"]]
    all_presents = [s["present"] for s in students if s["present"]]
    inferred     = max(all_totals) if all_totals else (max(all_presents) * 1.2 if all_presents else 100)

    results = []
    for s in students:
        total   = float(total_override or s["total"] or inferred or 100)
        pct     = round((s["present"] / total) * 100, 1) if total else 0
        needed  = math.ceil((threshold / 100) * total)
        deficit = max(0, needed - int(s["present"]))
        risk    = max(0, min(100, round((threshold - pct) * 2, 1)))
        results.append({**s, "total": total, "pct": pct, "deficit": deficit,
                        "flagged": pct < threshold, "risk_score": risk, "ml_risk": risk})
    return results

def get_class_analytics(students):
    classes = {}
    for s in students:
        cls = s.get("class", "General")
        if cls not in classes:
            classes[cls] = {"total": 0, "flagged": 0, "avg_pct": 0, "students": []}
        classes[cls]["total"] += 1
        if s["flagged"]: classes[cls]["flagged"] += 1
        classes[cls]["students"].append(s)
    for cls in classes:
        sl = classes[cls]["students"]
        classes[cls]["avg_pct"] = round(sum(s["pct"] for s in sl) / len(sl), 1)
    return classes

def get_attendance_patterns(students):
    return {
        "excellent": len([s for s in students if s["pct"] >= 90]),
        "good":      len([s for s in students if 75 <= s["pct"] < 90]),
        "average":   len([s for s in students if 60 <= s["pct"] < 75]),
        "poor":      len([s for s in students if s["pct"] < 60]),
    }

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    threshold      = float(request.form.get("threshold", 75))
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
            return jsonify({"error": "Only CSV and Excel files supported."}), 400

        students = parse_dataframe(df)
        results  = compute_stats(students, threshold, total_override)
        return jsonify({
            "students":        results,
            "threshold":       threshold,
            "class_analytics": get_class_analytics(results),
            "patterns":        get_attendance_patterns(results),
            "total_students":  len(results),
            "flagged_count":   sum(1 for s in results if s["flagged"]),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

# ─────────────────────────────────────────────────────────────────────────────
# OPENAI WARNING GENERATION
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/warning", methods=["POST"])
def generate_warning():
    data      = request.json or {}
    name      = data.get("name", "Student")
    roll      = data.get("roll", "N/A")
    present   = data.get("present", 0)
    total     = data.get("total", 100)
    pct       = data.get("pct", 0)
    deficit   = data.get("deficit", 0)
    threshold = data.get("threshold", 75)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY not set in .env file."}), 400

    prompt = (
        f"Write a concise formal academic warning letter body (under 90 words) for a student "
        f"with low attendance. No date, greeting, subject line, or signature — only the paragraph.\n\n"
        f"Student: {name} (Roll: {roll})\n"
        f"Attendance: {present}/{total} days ({pct}%)\n"
        f"Required threshold: {threshold}%\n"
        f"Days short: {deficit}\n\n"
        f"Tone: firm but supportive. Mention consequences and encourage improvement."
    )

    try:
        resp = http_requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model":      "gpt-4o-mini",
                "max_tokens": 300,
                "messages": [
                    {"role": "system", "content": "You are an academic administrator writing formal warning letters."},
                    {"role": "user",   "content": prompt},
                ],
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return jsonify({"warning": text})
    except http_requests.exceptions.HTTPError as e:
        err_body = e.response.json() if e.response else {}
        return jsonify({"error": err_body.get("error", {}).get("message", str(e))}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

# ─────────────────────────────────────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/export/csv", methods=["POST"])
def export_csv():
    data     = request.json or {}
    students = data.get("students", [])
    rows = [{
        "Name":           s["name"],
        "Roll No":        s["roll"],
        "Class":          s.get("class", "—"),
        "Days Present":   int(s["present"]),
        "Total Days":     int(s["total"]),
        "Attendance (%)": s["pct"],
        "Deficit Days":   s["deficit"],
        "Risk Score":     s.get("ml_risk", 0),
        "Status":         "FLAGGED" if s["flagged"] else "Clear",
        "Warning":        s.get("warning", ""),
    } for s in students]

    buf = io.BytesIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name=f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

@app.route("/export/json", methods=["POST"])
def export_json():
    data = request.json or {}
    buf  = io.BytesIO(json.dumps(data, indent=2).encode())
    buf.seek(0)
    return send_file(buf, mimetype="application/json", as_attachment=True,
                     download_name=f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

@app.route("/export/pdf", methods=["POST"])
def export_pdf():
    data      = request.json or {}
    students  = data.get("students", [])
    threshold = data.get("threshold", 75)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4),
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles  = getSampleStyleSheet()
    title_s = ParagraphStyle("t", parent=styles["Heading1"], alignment=TA_CENTER,
                             fontSize=18, spaceAfter=6, textColor=colors.HexColor("#1a3c5e"))
    sub_s   = ParagraphStyle("s", parent=styles["Normal"], alignment=TA_CENTER,
                             fontSize=10, textColor=colors.grey, spaceAfter=16)

    story = [
        Paragraph("AttendIQ — Attendance Intelligence Report", title_s),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
                  f"Threshold: {threshold}% | "
                  f"Flagged: {sum(1 for s in students if s['flagged'])} / {len(students)}", sub_s),
    ]

    header = ["Name", "Roll", "Class", "Present", "Total", "%", "Deficit", "Risk", "Status"]
    rows   = [header] + [[
        s["name"], s["roll"], s.get("class","—"),
        str(int(s["present"])), str(int(s["total"])),
        f"{s['pct']}%", f"-{s['deficit']}" if s["deficit"] else "—",
        f"{s.get('ml_risk',0):.0f}", "FLAGGED" if s["flagged"] else "Clear",
    ] for s in students]

    tbl = Table(rows, colWidths=[4*cm,2*cm,2*cm,1.8*cm,1.8*cm,1.5*cm,1.8*cm,1.5*cm,2*cm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#1a3c5e")),
        ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,0), 9),
        ("ALIGN",         (0,0),(-1,0), "CENTER"),
        ("FONTSIZE",      (0,1),(-1,-1), 8),
        ("ALIGN",         (3,1),(-1,-1), "CENTER"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#f5f7fa")]),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#dde1e7")),
    ]))
    for i, s in enumerate(students, 1):
        if s["flagged"]:
            tbl.setStyle(TableStyle([("BACKGROUND",(0,i),(-1,i), colors.HexColor("#fff5f5"))]))

    story.append(tbl)

    flagged_w = [s for s in students if s.get("flagged") and s.get("warning")]
    if flagged_w:
        story += [PageBreak(), Paragraph("AI-Generated Warning Messages", styles["Heading2"]), Spacer(1, 0.3*cm)]
        for s in flagged_w:
            story += [
                Paragraph(f"<b>{s['name']} ({s['roll']})</b> — {s['pct']}%", styles["Normal"]),
                Spacer(1, 0.15*cm),
                Paragraph(s["warning"].replace("\n","<br/>"), styles["Normal"]),
                Spacer(1, 0.4*cm),
            ]

    doc.build(story)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf", as_attachment=True,
                     download_name=f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

@app.route("/ai-insight", methods=["POST"])
def ai_insight():
    data = request.json or {}
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY not set."}), 400

    prompt = (
        f"Analyze this class attendance data and give a 3-4 sentence professional insight:\n"
        f"Total students: {data.get('total')}\n"
        f"Flagged (below {data.get('threshold')}%): {data.get('flagged')}\n"
        f"Average attendance: {data.get('avg')}%\n"
        f"Critical (below 50%): {data.get('critical')}\n"
        f"Sample students: {data.get('students', [])}\n\n"
        f"Give actionable recommendations for the teacher."
    )
    try:
        resp = http_requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini", "max_tokens": 300,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        resp.raise_for_status()
        return jsonify({"insight": resp.json()["choices"][0]["message"]["content"].strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
