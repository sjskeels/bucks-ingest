import os
import hashlib
import logging
from datetime import datetime, timezone

from flask import Flask, request, jsonify

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bucks-ingest")

INSERT_SQL = """
INSERT INTO raw.inbound_files (
  received_at,
  mail_from,
  mail_subject,
  mail_message_id,
  filename,
  content_type,
  file_bytes,
  sha256
)
VALUES (
  %(received_at)s,
  %(mail_from)s,
  %(mail_subject)s,
  %(mail_message_id)s,
  %(filename)s,
  %(content_type)s,
  %(file_bytes)s,
  %(sha256)s
)
ON CONFLICT (sha256) DO NOTHING
RETURNING id;
"""

def get_conn():
    # Lazy import so local /health runs without DB deps installed.
    import psycopg
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg.connect(db_url)

def create_app():
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return {"ok": True}, 200

    @app.post("/mailgun/inbound")
    def mailgun_inbound():
        mail_from = request.form.get("sender") or request.form.get("from") or ""
        subject = request.form.get("subject") or ""
        message_id = request.form.get("Message-Id") or request.form.get("message-id") or ""

        logger.info(
            "Inbound email: from=%r subject=%r message_id=%r form_keys=%s file_keys=%s",
            mail_from, subject, message_id,
            sorted(list(request.form.keys())),
            sorted(list(request.files.keys()))
        )

        # If we’re running locally without DATABASE_URL, fail clearly.
        if not os.getenv("DATABASE_URL"):
            return jsonify({"ok": False, "error": "DATABASE_URL not set (expected on Railway)"}), 500

        stored = 0
        deduped = 0
        results = []

        with get_conn() as conn:
            with conn.cursor() as cur:
                for key in request.files:
                    f = request.files.get(key)
                    if not f:
                        continue

                    filename = f.filename or "unknown"
                    content_type = f.content_type or "application/octet-stream"
                    file_bytes = f.read()

                    if not file_bytes or len(file_bytes) < 100:
                        logger.warning("Skipping tiny/empty file: key=%s filename=%s bytes=%s",
                                       key, filename, len(file_bytes) if file_bytes else 0)
                        continue

                    sha256 = hashlib.sha256(file_bytes).hexdigest()

                    payload = {
                        "received_at": datetime.now(timezone.utc),
                        "mail_from": mail_from,
                        "mail_subject": subject,
                        "mail_message_id": message_id,
                        "filename": filename,
                        "content_type": content_type,
                        "file_bytes": file_bytes,
                        "sha256": sha256,
                    }

                    cur.execute(INSERT_SQL, payload)
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        stored += 1
                        results.append({"key": key, "filename": filename, "bytes": len(file_bytes), "sha256": sha256, "id": row[0]})
                        logger.info("Stored: id=%s filename=%s bytes=%s sha256=%s", row[0], filename, len(file_bytes), sha256)
                    else:
                        deduped += 1
                        results.append({"key": key, "filename": filename, "bytes": len(file_bytes), "sha256": sha256, "deduped": True})
                        logger.info("Deduped: filename=%s sha256=%s", filename, sha256)

        return jsonify({"ok": True, "stored": stored, "deduped": deduped, "files": results}), 200

    return app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
