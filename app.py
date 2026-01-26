import os
import hashlib
import re
from datetime import datetime, date

from flask import Flask, request, jsonify
from psycopg import connect

app = Flask(__name__)


# --- Report metadata extraction (MVP) ---
# Goal: populate report_type + business_date WITHOUT parsing PDF contents.
# We infer from filename and/or email subject.

_REPORT_PREFIX_MAP = {
    # filename prefix -> report_type
    "salesstatreport": "salesstat",
    "salesclass": "salesclass",
}


def infer_report_type(filename: str | None, subject: str | None) -> str | None:
    fn = (filename or "").lower()
    sub = (subject or "").lower()

    for prefix, rtype in _REPORT_PREFIX_MAP.items():
        if fn.startswith(prefix):
            return rtype

    # subject-based fallback (cheap wins)
    if "daily sales statistics" in sub:
        return "salesstat"

    return None


def infer_business_date(filename: str | None, subject: str | None) -> date | None:
    """
    Try to infer business date from common patterns.
    Example filename: salesstatreport012526020108.PDF
      -> MMDDYY = 01/25/26
    """
    fn = (filename or "")

    # Pattern: <prefix><MMDDYY>...
    m = re.search(r"(salesstatreport|salesclass)(\d{6})", fn, flags=re.IGNORECASE)
    if m:
        mmddyy = m.group(2)
        try:
            return datetime.strptime(mmddyy, "%m%d%y").date()
        except ValueError:
            pass

    # Optional subject patterns: "... 2026-01-25" or "01/25/2026"
    sub = (subject or "")
    m = re.search(r"(\d{4}-\d{2}-\d{2})", sub)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass

    m = re.search(r"(\d{2}/\d{2}/\d{4})", sub)
    if m:
        try:
            return datetime.strptime(m.group(1), "%m/%d/%Y").date()
        except ValueError:
            pass

    return None


def get_db_url() -> str:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set (expected on Railway)")
    return db_url


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/mailgun/inbound")
def mailgun_inbound():
    sender = request.form.get("sender", "") or request.form.get("from", "")
    subject = request.form.get("subject", "")
    message_id = request.form.get("message-id", "") or request.form.get("Message-Id", "")

    files = request.files
    app.logger.info(
        "Inbound email: from=%r subject=%r message_id=%r form_keys=%s file_keys=%s",
        sender,
        subject,
        message_id,
        list(request.form.keys()),
        list(files.keys()),
    )

    stored = []

    # Iterate attachments: Mailgun uses attachment-1, attachment-2, ...
    for key in files:
        f = files[key]
        if not f:
            continue

        filename = f.filename or "attachment.bin"
        content_type = f.content_type or "application/octet-stream"
        blob = f.read() or b""
        sha256 = hashlib.sha256(blob).hexdigest()

        report_type = infer_report_type(filename, subject)
        business_date = infer_business_date(filename, subject)

        with connect(get_db_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO raw.inbound_files
                      (received_at, mail_from, mail_subject, mail_message_id,
                       filename, content_type, file_bytes, sha256, report_type, business_date)
                    VALUES (now(), %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (sha256) DO UPDATE
                      SET report_type   = COALESCE(raw.inbound_files.report_type, EXCLUDED.report_type),
                          business_date = COALESCE(raw.inbound_files.business_date, EXCLUDED.business_date)
                    RETURNING id
                    """,
                    (sender, subject, message_id, filename, content_type, blob, sha256, report_type, business_date),
                )
                row = cur.fetchone()
                new_id = row[0] if row else None

        app.logger.info(
            "Stored: id=%s filename=%s bytes=%s sha256=%s report_type=%s business_date=%s",
            new_id,
            filename,
            len(blob),
            sha256,
            report_type,
            business_date,
        )

        stored.append(
            {
                "id": new_id,
                "filename": filename,
                "bytes": len(blob),
                "sha256": sha256,
                "report_type": report_type,
                "business_date": str(business_date) if business_date else None,
            }
        )

    return jsonify({"ok": True, "stored": stored})
