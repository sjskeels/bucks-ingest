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

# -----------------------------------------------------------------------------
# Report metadata inference (MVP)
# -----------------------------------------------------------------------------

_REPORT_PREFIX_MAP = {
    "salesstatreport": "salesstat",
    "salesclassreport": "salesclass",
    "arinvregreport": "paytype",
}

def infer_report_type(filename: str | None, subject: str | None) -> str | None:
    fn = (filename or "").lower()
    sub = (subject or "").lower()

    for prefix, rtype in _REPORT_PREFIX_MAP.items():
        if fn.startswith(prefix):
            return rtype

    # subject-based fallbacks
    if "daily sales statistics" in sub:
        return "salesstat"
    if "sales by class" in sub or "sales class" in sub:
        return "salesclass"
    if "daily invoice list" in sub or "invoice list summary" in sub:
        return "paytype"

    return None


def infer_business_date(filename: str | None, subject: str | None) -> date | None:
    """
    Try to infer business date from common patterns.
    Examples:
      salesstatreport012526020108.PDF  -> MMDDYY = 01/25/26
      salesclassreport012726010024.PDF -> MMDDYY = 01/27/26
      arinvregreport012626010200.PDF   -> MMDDYY = 01/26/26
    """
    fn = (filename or "")

    m = re.search(r"(salesstatreport|salesclassreport|arinvregreport)(\d{6})", fn, flags=re.IGNORECASE)
    if m:
        mmddyy = m.group(2)
        try:
            return datetime.strptime(mmddyy, "%m%d%y").date()
        except ValueError:
            pass

    sub = (subject or "")

    # YYYY-MM-DD
    m = re.search(r"(\d{4}-\d{2}-\d{2})", sub)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass

    # MM/DD/YYYY
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


# -----------------------------------------------------------------------------
# Common parsing helpers
# -----------------------------------------------------------------------------

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


def _read_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    parts: list[str] = []
    for p in reader.pages:
        parts.append(p.extract_text() or "")
    return "\n".join(parts)


def _normalize_lines(text: str) -> list[str]:
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln]


def _fix_concatenated_decimals(s: str) -> str:
    # Example: "59.4564.431.27" -> "59.45 64.43 1.27"
    return re.sub(r"(\d\.\d{2})(?=\d)", r"\1 ", s)


# -----------------------------------------------------------------------------
# SalesStat parsing
# -----------------------------------------------------------------------------

def extract_salesstat_kpis(pdf_bytes: bytes) -> dict:
    """
    TransAct "Sales Statistics Report" extractor (MVP).

    Map TransAct labels -> our analytics columns:
      net_sales    <- "Sales SubTotal"
      gross_sales  <- "Total Sales and Tax"
      transactions <- "Number of Invoices" (or "Daily Invoices")
      avg_ticket   <- "Average Invoice Amount"
    """
    text = _read_pdf_text(pdf_bytes)
    lines = _normalize_lines(text)

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

    if sales_subtotal is None and total_sales_and_tax is None and avg_invoice_amount is None and invoices is None:
        sample = "\n".join(lines[:80])
        raise ValueError(f"Could not extract KPIs from PDF text. Sample:\n{sample}")

    return {
        "net_sales": sales_subtotal,
        "gross_sales": total_sales_and_tax,
        "transactions": invoices,
        "avg_ticket": avg_invoice_amount,
    }


# -----------------------------------------------------------------------------
# SalesClass parsing
# -----------------------------------------------------------------------------

def extract_salesclass_rows(pdf_bytes: bytes) -> list[dict]:
    """
    TransAct "Sales by Class" daily report.

    We extract per-class:
      class_code, class_name, net_sales, cost, gross_profit, margin_pct
    """
    text = _read_pdf_text(pdf_bytes)
    raw_lines = [ln for ln in text.splitlines() if ln.strip()]

    fixed_lines: list[str] = []
    for ln in raw_lines:
        ln = re.sub(r"\s+", " ", ln).strip()
        ln = _fix_concatenated_decimals(ln)
        ln = re.sub(r"(\d)([A-Za-z])", r"\1 \2", ln)  # "4.98Sporting" -> "4.98 Sporting"
        ln = re.sub(r"\s+", " ", ln).strip()
        fixed_lines.append(ln)

    # Example line after fixing:
    # 05 59.45 29.48 29.96 50.41% 4.98 Sporting Goods 59.45 64.43 1.27
    pat = re.compile(
        r"^(?P<class_code>(?:[A-Z0-9]{1,6}|<BLANK>))\s+"
        r"(?P<net>-?\d[\d,]*\.\d{2})\s+"
        r"(?P<cost>-?\d[\d,]*\.\d{2})\s+"
        r"(?P<gp>-?\d[\d,]*\.\d{2})\s+"
        r"(?P<margin>-?\d[\d,]*\.\d{2})%\s+"
        r"(?P<tax>-?\d[\d,]*\.\d{2})\s+"
        r"(?P<name>.+?)\s+"
        r"(?P<taxable>-?\d[\d,]*\.\d{2})\s+"
        r"(?P<gross>-?\d[\d,]*\.\d{2})\s+"
        r"(?P<tax2>-?\d[\d,]*\.\d{2})$",
        re.IGNORECASE,
    )

    rows: list[dict] = []
    for ln in fixed_lines:
        m = pat.match(ln)
        if not m:
            continue
        gd = m.groupdict()
        rows.append(
            {
                "class_code": gd["class_code"].strip(),
                "class_name": gd["name"].strip(),
                "net_sales": _money_to_decimal(gd["net"]),
                "cost": _money_to_decimal(gd["cost"]),
                "gross_profit": _money_to_decimal(gd["gp"]),
                "margin_pct": _money_to_decimal(gd["margin"]),
            }
        )

    if not rows:
        sample = "\n".join(fixed_lines[:120])
        raise ValueError(f"Could not extract any salesclass rows. Sample:\n{sample}")

    return rows


# -----------------------------------------------------------------------------
# PayType parsing (Invoice List Summary ordered by Payment Type)
# -----------------------------------------------------------------------------

def extract_paytype_totals(pdf_bytes: bytes) -> list[dict]:
    """
    Extract per payment type totals from the 'Invoice List Summary' report.

    The report prints subtotal lines like:
      1,187.9530.19Acct ChgPayment Type: 797.24 360.52
    Meaning:
      invoice_total = 1187.95
      tax_amount    = 30.19
      payment_type  = "Acct Chg"
      taxed_sales   = 797.24
      nontaxed_sales= 360.52

    Sanity check:
      taxed_sales + nontaxed_sales == invoice_total - tax_amount
    """
    text = _read_pdf_text(pdf_bytes)
    blob = re.sub(r"\s+", " ", text).strip()

    # Regex that tolerates missing spaces (e.g., "...30.19Acct ChgPayment Type...")
    pat = re.compile(
        r"(?P<invoice_total>-?\d[\d,]*\.\d{2})\s*"
        r"(?P<tax_amount>-?\d[\d,]*\.\d{2})\s*"
        r"(?P<payment_type>[A-Za-z0-9/ ]{1,20}?)(?=Payment\s*Type:)\s*"
        r"Payment\s*Type:\s*"
        r"(?P<taxed_sales>-?\d[\d,]*\.\d{2})\s*"
        r"(?P<nontaxed_sales>-?\d[\d,]*\.\d{2})",
        re.IGNORECASE,
    )

    matches = list(pat.finditer(blob))
    if not matches:
        sample = blob[:2000]
        raise ValueError(f"Could not extract any payment type totals. Sample:\n{sample}")

    # last occurrence per payment type wins (avoids accidental partials)
    out_by_type: dict[str, dict] = {}

    for m in matches:
        gd = m.groupdict()
        ptype = re.sub(r"\s+", " ", (gd["payment_type"] or "").strip())
        inv_total = _money_to_decimal(gd["invoice_total"]) or Decimal("0")
        tax_amt = _money_to_decimal(gd["tax_amount"]) or Decimal("0")
        taxed = _money_to_decimal(gd["taxed_sales"]) or Decimal("0")
        nontaxed = _money_to_decimal(gd["nontaxed_sales"]) or Decimal("0")

        # Sanity check: if swapped, fix it
        expected_sales = inv_total - tax_amt
        if (taxed + nontaxed - expected_sales).copy_abs() > Decimal("0.05"):
            if (nontaxed + taxed - expected_sales).copy_abs() <= Decimal("0.05"):
                taxed, nontaxed = nontaxed, taxed

        out_by_type[ptype] = {
            "payment_type": ptype,
            "invoice_total": inv_total,
            "tax_amount": tax_amt,
            "taxed_sales": taxed,
            "nontaxed_sales": nontaxed,
            "amount": taxed + nontaxed,  # sales excluding tax
            "txn_count": None,           # not reliably available in this report
        }

    return list(out_by_type.values())


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Parse endpoints: SalesStat
# -----------------------------------------------------------------------------

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
                        kpis["net_sales"],
                        kpis["gross_sales"],
                        kpis["transactions"],
                        kpis["avg_ticket"],
                    ),
                )
                out = cur.fetchone()

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
        biz_date = datetime.strptime(business_date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"ok": False, "error": "invalid business_date (expected YYYY-MM-DD)"}), 400

    try:
        with connect(get_db_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, business_date, filename, sha256, file_bytes
                    from raw.inbound_files
                    where report_type = 'salesstat'
                      and business_date = %s
                    order by received_at desc
                    limit 1
                    """,
                    (biz_date,),
                )
                row = cur.fetchone()
                if not row:
                    return jsonify({"ok": False, "error": f"no salesstat PDF found for {business_date_str}"}), 404

                file_id, biz_date, filename, sha256, pdf_bytes = row
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
                        kpis["net_sales"],
                        kpis["gross_sales"],
                        kpis["transactions"],
                        kpis["avg_ticket"],
                    ),
                )
                out = cur.fetchone()

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


# -----------------------------------------------------------------------------
# Parse endpoints: SalesClass
# -----------------------------------------------------------------------------

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
                    select id, business_date, filename, sha256, file_bytes
                    from raw.inbound_files
                    where report_type = 'salesclass'
                    order by received_at desc
                    limit 1
                    """
                )
                row = cur.fetchone()
                if not row:
                    return jsonify({"ok": False, "error": "No salesclass PDFs found in raw.inbound_files"}), 404

                file_id, biz_date, filename, sha256, pdf_bytes = row
                if not biz_date:
                    return jsonify({"ok": False, "error": "Latest salesclass PDF missing business_date"}), 400

                rows = extract_salesclass_rows(pdf_bytes)

                # Upsert each class row
                upserted = 0
                for r in rows:
                    cur.execute(
                        """
                        insert into analytics.salesclass_daily
                          (business_date, class_code, class_name,
                           net_sales, cost, gross_profit, margin_pct,
                           source_file_id, source_sha256, updated_at)
                        values
                          (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                        on conflict (business_date, class_code) do update
                          set class_name   = excluded.class_name,
                              net_sales    = excluded.net_sales,
                              cost         = excluded.cost,
                              gross_profit = excluded.gross_profit,
                              margin_pct   = excluded.margin_pct,
                              source_file_id = excluded.source_file_id,
                              source_sha256  = excluded.source_sha256,
                              updated_at   = now()
                        """,
                        (
                            biz_date,
                            r["class_code"],
                            r["class_name"],
                            r["net_sales"],
                            r["cost"],
                            r["gross_profit"],
                            r["margin_pct"],
                            file_id,
                            sha256,
                        ),
                    )
                    upserted += 1

        return jsonify(
            {
                "ok": True,
                "source": {"id": file_id, "filename": filename, "sha256": sha256, "business_date": str(biz_date)},
                "upserted_rows": upserted,
            }
        )
    except Exception as e:
        app.logger.exception("parse_salesclass_latest failed")
        return jsonify({"ok": False, "error_type": type(e).__name__, "error": str(e)}), 500


# -----------------------------------------------------------------------------
# Parse endpoints: PayType
# -----------------------------------------------------------------------------

@app.post("/parse/paytype/latest")
def parse_paytype_latest():
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
                    where report_type = 'paytype'
                    order by received_at desc
                    limit 1
                    """
                )
                row = cur.fetchone()
                if not row:
                    return jsonify({"ok": False, "error": "No paytype PDFs found in raw.inbound_files"}), 404

                file_id, biz_date, filename, sha256, pdf_bytes = row
                if not biz_date:
                    return jsonify({"ok": False, "error": "Latest paytype PDF missing business_date"}), 400

                totals = extract_paytype_totals(pdf_bytes)

                upserted = 0
                for t in totals:
                    cur.execute(
                        """
                        insert into analytics.paytype_daily
                          (business_date, payment_type,
                           amount, txn_count,
                           taxed_sales, nontaxed_sales, tax_amount, invoice_total,
                           source_file_id, source_sha256, updated_at)
                        values
                          (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                        on conflict (business_date, payment_type) do update
                          set amount        = excluded.amount,
                              txn_count     = excluded.txn_count,
                              taxed_sales   = excluded.taxed_sales,
                              nontaxed_sales= excluded.nontaxed_sales,
                              tax_amount    = excluded.tax_amount,
                              invoice_total = excluded.invoice_total,
                              source_file_id = excluded.source_file_id,
                              source_sha256  = excluded.source_sha256,
                              updated_at    = now()
                        """,
                        (
                            biz_date,
                            t["payment_type"],
                            t["amount"],
                            t["txn_count"],
                            t["taxed_sales"],
                            t["nontaxed_sales"],
                            t["tax_amount"],
                            t["invoice_total"],
                            file_id,
                            sha256,
                        ),
                    )
                    upserted += 1

        return jsonify(
            {
                "ok": True,
                "source": {"id": file_id, "filename": filename, "sha256": sha256, "business_date": str(biz_date)},
                "upserted_rows": upserted,
            }
        )
    except Exception as e:
        app.logger.exception("parse_paytype_latest failed")
        return jsonify({"ok": False, "error_type": type(e).__name__, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
