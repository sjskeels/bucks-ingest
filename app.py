import os
import hashlib
import re
from datetime import datetime
from flask import Flask, request, abort
import psycopg2
import requests

app = Flask(__name__)

# --- helpers -------------------------------------------------

def db_conn():
    """
    Railway Postgres: use DATABASE_URL if present.
    """
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set in Railway variables.")
    return psycopg2.connect(dsn)

def guess_report_type(filename: str, subject: str) -> str:
    """
    MVP heuristic ONLY. We will refine later.
    """
    s = f"{filename} {subject}".lower()

    # common TransAct patterns / your report names
    if "salesstat" in s or "daily sales statistic" in s or "daily sales statistics" in s:
        return "salesstat"
    if "arinvreg" in s or "daily invoice list" in s or "invoice list" in s:
        return "arinvreg"
    if "salesclass" in s or "sales by class" in s:
        return "salesclass"

    return "unknown"

def extract_business_date(subject: str, body_text: str) -> str | None:
    """
    Best-effort: try to find a date like 01/12/26 or 2026-01-12.
    Returns ISO date string 'YYYY-MM-DD' or None.
    """
    hay = f"{subject}\n{body_text}"

    # match mm/dd/yy
    m = re.search(r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(?:\d{2})\b", hay)
    if m:
        mm, dd, yy = m.group(0).split("/")
        yy = int(yy)
        yy += 2000 if yy < 70 else 1900
        dt = datetime(yy, int(mm), int(dd))
        return dt.strftime("%Y-%m-%d")

    # match yyyy-mm-dd
    m = re.search(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b", hay)
    if m:
        return m.group(0)

    return None

# --- routes ---------------------------------------------------

@app.get("/")
def health():
    return {"ok": True}

@app.post("/mailgun/inbound")
def inbound():
    """
    Mailgun routes -> forward to this endpoint.
    We store each PDF attachment as a row in raw.inbound_files.
    """
    # Optional shared secret (recommended). If set, require it.
    expected = os.environ.get("INGEST_SECRET")
    if expected:
        got = request.headers.get("X-Ingest-Secret") or request.args.get("secret")
        if got != expected:
            abort(401, "unauthorized")

    mail_from = request.form.get("from") or request.form.get("sender")
    mail_subject = request.form.get("subject")
    mail_message_id = request.form.get("Message-Id") or request.form.get("message-id")

    # Find attachments (Mailgun sends as files: attachment-1, attachment-2, ...)
    attachments = []
    for key in request.files:
        if key.startswith("attachment-"):
            attachments.append((key, request.files[key]))

    if not attachments:
        # Nothing to ingest, but don't fail Mailgun
        return {"ok": True, "ingested": 0, "reason": "no attachments"}, 200

    ingested = 0

    with db_conn() as conn:
        with conn.cursor() as cur:
            for key, f in attachments:
                filename = f.filename or "attachment.pdf"
                content_type = f.mimetype or "application/octet-stream"
                file_bytes = f.read()

                if not file_bytes:
                    continue

                sha256 = hashlib.sha256(file_bytes).hexdigest()

                # Small text fields that sometimes help infer date/type
                stripped_text = (request.form.get("stripped-text") or "")[:10000]
                business_date = extract_business_date(mail_subject or "", stripped_text)

                report_type = guess_report_type(filename, mail_subject or "")

                cur.execute(
                    """
                    INSERT INTO raw.inbound_files
                      (mail_from, mail_subject, mail_message_id,
                       filename, content_type, file_bytes, sha256, report_type, business_date)
                    VALUES
                      (%s, %s, %s,
                       %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (sha256) DO NOTHING
                    """,
                    (
                        mail_from,
                        mail_subject,
                        mail_message_id,
                        filename,
                        content_type,
                        psycopg2.Binary(file_bytes),
                        sha256,
                        report_type,
                        business_date,
                    ),
                )
                ingested += 1

    return {"ok": True, "ingested": ingested}, 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
