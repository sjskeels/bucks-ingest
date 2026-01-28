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
    "salesclassreport": "salesclass",
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
    Example filenames:
      salesstatreport012526020108.PDF  -> MMDDYY=012526 -> 2026-01-25
      salesclassreport012726010024.PDF -> MMDDYY=012726 -> 2026-01-27
    """
    fn = filename or ""

    m = re.search(r"(salesstatreport|salesclassreport)(\d{6})", fn, flags=re.IGNORECASE)
    if m:
        mmddyy = m.group(2)
        try:
            return datetime.strptime(mmddyy, "%m%d%y").date()
        except ValueError:
            pass

    sub = subject or ""

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


# -----------------------
# SalesStat extractor
# -----------------------
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


def _upsert_salesstat_daily(cur, biz_date: date, kpis: dict):
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
    return cur.fetchone()


# -----------------------
# SalesClass extractor
# -----------------------
_salesclass_num_re = re.compile(r"-?\d[\d,]*\.\d{2}%?")


def extract_salesclass_rows(pdf_bytes: bytes) -> list[dict]:
    """
    TransAct "Sales by Class Summary Report" extractor.

    Columns we store (per class_code):
      ext_price, ext_cost, profit, gp_pct, ext_tax, price_plus_tax, pct_total, class_name
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    text = "\n".join([(p.extract_text() or "") for p in reader.pages])

    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    rows: list[dict] = []

    for line in lines:
        l = re.sub(r"(?<=\d)(?=[A-Za-z<])", " ", line)
        l = re.sub(r"(?<=%)(?=[A-Za-z])", " ", l)
        ll = l.lower()

        if "reports totals" in ll or ll.startswith("printed:") or ll.endswith("dssumcls"):
            continue

        parts = l.split()
        if not parts:
            continue

        class_code = parts[0]

        # Skip lines where the "class_code" is actually a money value
        if re.match(r"^\d[\d,]*\.\d{2}$", class_code):
            continue

        matches = list(_salesclass_num_re.finditer(l))
        if len(matches) < 5:
            continue

        # First five numeric tokens
        ext_price = matches[0].group()
        ext_cost = matches[1].group()
        profit = matches[2].group()
        gp_pct = matches[3].group().replace("%", "")
        ext_tax = matches[4].group()

        # Remaining numeric tokens (typically: repeated ext_price, price_plus_tax, pct_total)
        post = [m.group() for m in matches[5:]]
        price_plus_tax = post[-2] if len(post) >= 2 else None
        pct_total = post[-1] if len(post) >= 1 else None

        class_name = None
        if len(matches) >= 6:
            class_name = l[matches[4].end() : matches[5].start()].strip()
            class_name = class_name if class_name else None

        rows.append(
            {
                "class_code": class_code,
                "class_name": class_name,
                "ext_price": ext_price,
                "ext_cost": ext_cost,
                "profit": profit,
                "gp_pct": gp_pct,
                "ext_tax": ext_tax,
                "price_plus_tax": price_plus_tax,
                "pct_total": pct_total,
            }
        )

    if not rows:
        sample = "\n".join(lines[:120])
        raise ValueError(f"Could not extract any SalesClass rows. Sample:\n{sample}")

    return rows


def _upsert_salesclass_rows(cur, biz_date: date, source_file_id: int, source_sha256: str, rows: list[dict]) -> int:
    n = 0
    for r in rows:
        cur.execute(
            """
            insert into analytics.salesclass_daily
              (business_date, class_code, class_name,
               ext_price, pct_total, ext_cost, profit, gp_pct, ext_tax, price_plus_tax,
               source_file_id, source_sha256, updated_at)
            values
              (%s, %s, %s,
               %s, %s, %s, %s, %s, %s, %s,
               %s, %s, now())
            on conflict (business_date, class_code) do update
              set class_name      = excluded.class_name,
                  ext_price       = excluded.ext_price,
                  pct_total       = excluded.pct_total,
                  ext_cost        = excluded.ext_cost,
                  profit          = excluded.profit,
                  gp_pct          = excluded.gp_pct,
                  ext_tax         = excluded.ext_tax,
                  price_plus_tax  = excluded.price_plus_tax,
                  source_file_id  = excluded.source_file_id,
                  source_sha256   = excluded.source_sha256,
                  updated_at      = now()
            """,
            (
                biz_date,
                r["class_code"],
                r["class_name"],
                Decimal(r["ext_price"]) if r["ext_price"] is not None else None,
                Decimal(r["pct_total"]) if r["pct_total"] is not None else None,
                Decimal(r["ext_cost"]) if r["ext_cost"] is not None else None,
                Decimal(r["profit"]) if r["profit"] is not None else None,
                Decimal(r["gp_pct"]) if r["gp_pct"] is not None else None,
                Decimal(r["ext_tax"]) if r["ext_tax"] is not None else None,
                Decimal(r["price_plus_tax"]) if r["price_plus_tax"] is not None else None,
                source_file_id,
                source_sha256,
            ),
        )
        n += 1
    return n


# -----------------------
# Flask routes
# -----------------------
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


# --- SalesStat endpoints ---
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
                    select id, business_date, filename, mail_subject, sha256, file_bytes
                    from raw.inbound_files
                    where report_type = 'salesstat'
                    order by received_at desc
                    limit 1
                    """
                )
                row = cur.fetchone()
                if not row:
                    return jsonify({"ok": False, "error": "No salesstat PDFs found in raw.inbound_files"}), 404

                file_id, biz_date, filename, mail_subject, sha256, pdf_bytes = row
                if not biz_date:
                    biz_date = infer_business_date(filename, mail_subject)
                    if biz_date:
                        cur.execute("update raw.inbound_files set business_date=%s where id=%s", (biz_date, file_id))
                    else:
                        return jsonify({"ok": False, "error": "Latest salesstat PDF missing business_date"}), 400

                kpis = extract_salesstat_kpis(pdf_bytes)
                out = _upsert_salesstat_daily(cur, biz_date, kpis)

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


@app.post("/parse/salesstat/by-date")
def parse_salesstat_by_date():
    try:
        require_parse_token()
    except PermissionError as e:
        return jsonify({"ok": False, "error": str(e)}), 401

    business_date_str = (request.args.get("business_date") or "").strip()
    if not business_date_str:
        return jsonify({"ok": False, "error": "missing business_date (YYYY-MM-DD)"}), 400

    try:
        target_date = datetime.strptime(business_date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"ok": False, "error": "invalid business_date (expected YYYY-MM-DD)"}), 400

    try:
        with connect(get_db_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, business_date, filename, mail_subject, sha256, file_bytes
                    from raw.inbound_files
                    where report_type='salesstat'
                      and business_date=%s
                    order by received_at desc
                    limit 1
                    """,
                    (target_date,),
                )
                row = cur.fetchone()
                if not row:
                    return jsonify({"ok": False, "error": f"No salesstat PDF found for {business_date_str}"}), 404

                file_id, biz_date, filename, mail_subject, sha256, pdf_bytes = row
                if not biz_date:
                    biz_date = infer_business_date(filename, mail_subject) or target_date

                kpis = extract_salesstat_kpis(pdf_bytes)
                out = _upsert_salesstat_daily(cur, biz_date, kpis)

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
        app.logger.exception("parse_salesstat_by_date failed")
        return jsonify({"ok": False, "error_type": type(e).__name__, "error": str(e)}), 500


# --- SalesClass endpoints ---
@app.post("/parse/salesclass/latest")
def parse_salesclass_latest():
    try:
        require_parse_token()
    except PermissionError as e:
        return jsonify({"ok": False, "error": str(e)}), 401

    try:
        with connect(get_db_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, business_date, filename, mail_subject, sha256, file_bytes
                    from raw.inbound_files
                    where report_type = 'salesclass'
                    order by received_at desc
                    limit 1
                    """
                )
                row = cur.fetchone()
                if not row:
                    return jsonify({"ok": False, "error": "No salesclass PDFs found in raw.inbound_files"}), 404

                file_id, biz_date, filename, mail_subject, sha256, pdf_bytes = row
                if not biz_date:
                    biz_date = infer_business_date(filename, mail_subject)
                    if biz_date:
                        cur.execute("update raw.inbound_files set business_date=%s where id=%s", (biz_date, file_id))
                    else:
                        return jsonify({"ok": False, "error": "Latest salesclass PDF missing business_date"}), 400

                rows = extract_salesclass_rows(pdf_bytes)
                n = _upsert_salesclass_rows(cur, biz_date, file_id, sha256, rows)

        return jsonify(
            {
                "ok": True,
                "source": {"id": file_id, "filename": filename, "sha256": sha256, "business_date": str(biz_date)},
                "rows_upserted": n,
            }
        )

    except Exception as e:
        app.logger.exception("parse_salesclass_latest failed")
        return jsonify({"ok": False, "error_type": type(e).__name__, "error": str(e)}), 500


@app.post("/parse/salesclass/by-date")
def parse_salesclass_by_date():
    try:
        require_parse_token()
    except PermissionError as e:
        return jsonify({"ok": False, "error": str(e)}), 401

    business_date_str = (request.args.get("business_date") or "").strip()
    if not business_date_str:
        return jsonify({"ok": False, "error": "missing business_date (YYYY-MM-DD)"}), 400

    try:
        target_date = datetime.strptime(business_date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"ok": False, "error": "invalid business_date (expected YYYY-MM-DD)"}), 400

    # Also match old rows that may have business_date NULL but filename contains MMDDYY
    mmddyy = target_date.strftime("%m%d%y")

    try:
        with connect(get_db_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, business_date, filename, mail_subject, sha256, file_bytes
                    from raw.inbound_files
                    where report_type='salesclass'
                      and (
                        business_date=%s
                        or (business_date is null and filename ilike %s)
                      )
                    order by received_at desc
                    limit 1
                    """,
                    (target_date, f"%{mmddyy}%"),
                )
                row = cur.fetchone()
                if not row:
                    return jsonify({"ok": False, "error": f"No salesclass PDF found for {business_date_str}"}), 404

                file_id, biz_date, filename, mail_subject, sha256, pdf_bytes = row
                if not biz_date:
                    biz_date = infer_business_date(filename, mail_subject) or target_date
                    cur.execute("update raw.inbound_files set business_date=%s where id=%s", (biz_date, file_id))

                rows = extract_salesclass_rows(pdf_bytes)
                n = _upsert_salesclass_rows(cur, biz_date, file_id, sha256, rows)

        return jsonify(
            {
                "ok": True,
                "source": {"id": file_id, "filename": filename, "sha256": sha256, "business_date": str(biz_date)},
                "rows_upserted": n,
            }
        )

    except Exception as e:
        app.logger.exception("parse_salesclass_by_date failed")
        return jsonify({"ok": False, "error_type": type(e).__name__, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
