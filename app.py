import os
import hashlib
import re
from datetime import datetime, date
from io import BytesIO
from decimal import Decimal

from flask import Flask, request, jsonify
from psycopg import connect
from pypdf import PdfReader

app = Flask(__name__)


# --- Report metadata extraction (MVP) ---
# Goal: populate report_type + business_date WITHOUT parsing PDF contents.
# We infer from filename and/or email subject.

_REPORT_PREFIX_MAP = {
    "salesstatreport": "salesstat",
    "salesclass": "salesclass",
}


def infer_report_type(filename: str | None, subject: str | None) -> str | None:
    fn = (filename or "").lower()
    sub = (subject or "").lower()

    for prefix, rtype in _REPORT_PREFIX_MAP.items():
        if fn.startswith(prefix):
            return rtype

    if "daily sales statistics" in sub:
        return "salesstat"

    return None


def infer_business_date(filename: str | None, subject: str | None) -> date | None:
    """
    Try to infer business date from common patterns.
    Example filename: salesstatreport012526020108.PDF -> MMDDYY = 01/25/26
    """
    fn = (filename or "")

    m = re.search(r"(salesstatreport|salesclass)(\d{6})", fn, flags=re.IGNORECASE)
    if m:
        mmddyy = m.group(2)
        try:
            return datetime.strptime(mmddyy, "%m%d%y").date()
        except ValueError:
            pass

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


def require_parse_token() -> None:
    """
    Safety valve so parsing isn't public.
    Set PARSE_TOKEN on Railway. Call with header: X-Parse-Token: <token>
    If PARSE_TOKEN is unset, allow (local dev convenience).
    """
    expected = os.getenv("PARSE_TOKEN", "").strip()
    if not expected:
        return
    got = (request.headers.get("X-Parse-Token") or "").strip()
    if got != expected:
        raise PermissionError("Missing/invalid X-Parse-Token")


_money_re = re.compile(r"(\(?-?\$?\d[\d,]*\.\d{2}\)?)")
_int_re = re.compile(r"(\d[\d,]*)")


def _money_to_decimal(s: str) -> Decimal | None:
    if not s:
        return None
    s = s.strip()
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace("$", "").replace(",", "").strip()
    try:
        val = Decimal(s)
        return -val if neg else val
    except Exception:
        return None


def extract_salesstat_kpis(pdf_bytes: bytes) -> dict:
    """
    TransAct "Sales Statistics Report" extractor (MVP).

    Map TransAct labels -> our analytics columns:
      net_sales    <- "Sales SubTotal"
      gross_sales  <- "Total Sales and Tax"
      transactions <- "Number of Invoices" (or "Daily Invoices")
      avg_ticket   <- "Average Invoice Amount"
    """
    reader = PdfReader(BytesIO(pdf_bytes))

    text_parts = []
    for p in reader.pages:
        t = p.extract_text() or ""
        text_parts.append(t)
    text = "\n".join(text_parts)

    # Normalize lines
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    def find_money(label_variants: list[str]) -> Decimal | None:
        for ln in lines:
            lnl = ln.lower()
            if any(lbl in lnl for lbl in label_variants):
                m = _money_re.findall(ln)
                if m:
                    return _money_to_decimal(m[-1])
        return None

    def find_int(label_variants: list[str]) -> int | None:
        for ln in lines:
            lnl = ln.lower()
            if any(lbl in lnl for lbl in label_variants):
                m = _int_re.findall(ln)
                if m:
                    try:
                        return int(m[-1].replace(",", ""))
                    except Exception:
                        return None
        return None

    sales_subtotal = find_money(["sales subtotal"])
    total_sales_and_tax = find_money(["total sales and tax"])
    avg_invoice_amount = find_money(["average invoice amount"])

    invoices = find_int(["number of invoices", "daily invoices"])

    # If nothing found, throw a helpful error sample
    if (
        sales_subtotal is None
        and total_sales_and_tax is None
        and avg_invoice_amount is None
        and invoices is None
    ):
        sample = "\n".join(lines[:80])
        raise ValueError(f"Could not extract KPIs from PDF text. Sample:\n{sample}")

    return {
        "net_sales": str(sales_subtotal) if sales_subtotal is not None else None,
        "gross_sales": str(total_sales_and_tax) if total_sales_and_tax is not None else None,
        "transactions": invoices,
        "avg_ticket": str(avg_invoice_amount) if avg_invoice_amount is not None else None,
    }


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


@app.post("/parse/salesstat/latest")
def parse_salesstat_latest():
    try:
        require_parse_token()
    except PermissionError as e:
        return jsonify({"ok": False, "error": str(e)}), 401

    try:
        with connect(get_db_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, business_date, filename, sha256, file_bytes
                    from raw.inbound_files
                    where report_type = 'salesstat'
                    order by received_at desc
                    limit 1
                    """
                )
                row = cur.fetchone()
                if not row:
                    return jsonify({"ok": False, "error": "No salesstat PDFs found in raw.inbound_files"}), 404

                file_id, biz_date, filename, sha256, pdf_bytes = row
                if not biz_date:
                    return jsonify(
                        {
                            "ok": False,
                            "error": "Latest salesstat PDF missing business_date; cannot parse into daily table",
                            "id": file_id,
                        }
                    ), 400

                kpis = extract_salesstat_kpis(pdf_bytes)

                cur.execute(
                    """
                    insert into analytics.salesstat_daily
                      (business_date, net_sales, gross_sales, transactions, avg_ticket, updated_at)
                    values
                      (%s, %s, %s, %s, %s, now())
                    on conflict (business_date) do update
                      set net_sales = excluded.net_sales,
                          gross_sales = excluded.gross_sales,
                          transactions = excluded.transactions,
                          avg_ticket = excluded.avg_ticket,
                          updated_at = now()
                    returning business_date, net_sales, gross_sales, transactions, avg_ticket, updated_at
                    """,
                    (
                        biz_date,
                        Decimal(kpis["net_sales"]) if kpis["net_sales"] is not None else None,
                        Decimal(kpis["gross_sales"]) if kpis["gross_sales"] is not None else None,
                        kpis["transactions"],
                        Decimal(kpis["avg_ticket"]) if kpis["avg_ticket"] is not None else None,
                    ),
                )
                out = cur.fetchone()

        app.logger.info("Parsed salesstat: id=%s sha256=%s business_date=%s kpis=%s", file_id, sha256, biz_date, kpis)

        return jsonify(
            {
                "ok": True,
                "source": {"id": file_id, "filename": filename, "sha256": sha256, "business_date": str(biz_date)},
                "upserted": {
                    "business_date": str(out[0]),
                    "net_sales": str(out[1]) if out[1] is not None else None,
                    "gross_sales": str(out[2]) if out[2] is not None else None,
                    "transactions": out[3],
                    "avg_ticket": str(out[4]) if out[4] is not None else None,
                    "updated_at": out[5].isoformat() if out[5] is not None else None,
                },
            }
        )

    except Exception as e:
        app.logger.exception("parse_salesstat_latest failed")
        return jsonify({"ok": False, "error_type": type(e).__name__, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
