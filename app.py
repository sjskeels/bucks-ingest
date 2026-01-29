import os
import hashlib
import re
from datetime import datetime, date
from io import BytesIO
from decimal import Decimal
from collections import defaultdict

from flask import Flask, request, jsonify
from psycopg import connect
from pypdf import PdfReader

app = Flask(__name__)

# -----------------------------------------------------------------------------
# Report metadata extraction (MVP)
# Goal: populate report_type + business_date before parsing contents.
# -----------------------------------------------------------------------------

_REPORT_PREFIX_MAP = {
    "salesstatreport": "salesstat",
    "salesclassreport": "salesclass",
    "arinvregreport": "paytype",  # Daily Invoice List (grouped by payment type)
}

_SUBJECT_MAP = {
    "daily sales statistics": "salesstat",
    "daily sales by class": "salesclass",
    "daily invoice list": "paytype",
}

def infer_report_type(filename: str | None, subject: str | None) -> str | None:
    fn = (filename or "").lower()
    sub = (subject or "").lower()

    for prefix, rtype in _REPORT_PREFIX_MAP.items():
        if fn.startswith(prefix):
            return rtype

    for needle, rtype in _SUBJECT_MAP.items():
        if needle in sub:
            return rtype

    return None


def infer_business_date(filename: str | None, subject: str | None) -> date | None:
    """
    Infer business_date from common patterns.
    Example filenames:
      salesstatreport012526020108.PDF -> MMDDYY = 01/25/26
      salesclassreport012726010024.PDF -> MMDDYY = 01/27/26
      arinvregreport012726010235.PDF   -> MMDDYY = 01/27/26
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

    # 2026-01-27
    m = re.search(r"(\d{4}-\d{2}-\d{2})", sub)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass

    # 01/27/2026
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
# Helpers
# -----------------------------------------------------------------------------

_money = r"-?\d[\d,]*\.\d{2}"
_money_re = re.compile(rf"({_money})")

def _to_decimal(s: str | None) -> Decimal | None:
    if not s:
        return None
    try:
        return Decimal(s.replace(",", "").strip())
    except Exception:
        return None


_table_cols_cache: dict[tuple[str, str], set[str]] = {}

def _get_table_cols(cur, schema: str, table: str) -> set[str]:
    key = (schema, table)
    if key in _table_cols_cache:
        return _table_cols_cache[key]

    cur.execute(
        """
        select column_name
        from information_schema.columns
        where table_schema = %s
          and table_name   = %s
        """,
        (schema, table),
    )
    cols = {r[0] for r in cur.fetchall()}
    _table_cols_cache[key] = cols
    return cols


def _insert_rows(cur, schema: str, table: str, rows: list[dict]):
    """
    Insert rows into schema.table, automatically filtering to existing columns.
    Adds updated_at=now() if that column exists.
    """
    if not rows:
        return 0

    cols = _get_table_cols(cur, schema, table)
    if not cols:
        raise RuntimeError(f"Could not introspect columns for {schema}.{table}")

    # Determine columns to insert (intersection across rows + existing cols)
    # Keep stable order (explicit list)
    desired_order = [
        "business_date",
        "payment_type",
        "class_code",
        "class_name",
        "amount",
        "invoice_total",
        "tax_amount",
        "taxed_sales",
        "nontaxed_sales",
        "sales_total",
        "txn_count",
        "sales",
        "tax",
        "sales_plus_tax",
        "cost",
        "profit",
        "gp_pct",
        "pct_total",
        "source_file_id",
        "source_sha256",
        "sha256",
        "filename",
        "report_type",
        "updated_at",
    ]

    insert_cols = []
    for c in desired_order:
        if c == "updated_at":
            continue
        if c in cols and any(c in r for r in rows):
            insert_cols.append(c)

    has_updated_at = "updated_at" in cols

    # Build SQL
    col_sql = ", ".join(insert_cols + (["updated_at"] if has_updated_at else []))
    placeholders = ", ".join(["%s"] * len(insert_cols) + (["now()"] if has_updated_at else []))

    sql = f"insert into {schema}.{table} ({col_sql}) values ({placeholders})"

    values = []
    for r in rows:
        row_vals = []
        for c in insert_cols:
            row_vals.append(r.get(c))
        values.append(tuple(row_vals))

    cur.executemany(sql, values)
    return len(rows)


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
    reader = PdfReader(BytesIO(pdf_bytes))
    text = "\n".join((p.extract_text() or "") for p in reader.pages)

    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]

    def find_money(label_variants: list[str]) -> Decimal | None:
        for ln in lines:
            lnl = ln.lower()
            if any(lbl in lnl for lbl in label_variants):
                m = _money_re.findall(ln)
                if m:
                    return _to_decimal(m[-1])
        return None

    def find_int(label_variants: list[str]) -> int | None:
        for ln in lines:
            lnl = ln.lower()
            if any(lbl in lnl for lbl in label_variants):
                m = re.findall(r"(\d[\d,]*)", ln)
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
    Parse TransAct 'Daily Sales by Class' report.

    We store per class:
      class_code, class_name, sales (extended price), tax, sales_plus_tax, cost, profit, gp_pct, pct_total
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]

    money_iter_re = re.compile(_money)
    pct_re = re.compile(r"(\d[\d,]*\.\d{2})%")

    out = []
    for ln in lines:
        if ln.startswith(("Daily Sales by Class", "From:", "Time:", "Class ", "Note:", "Printed:")):
            continue

        m_code = re.match(r"^(?P<code>[A-Z0-9]{2,6})\s", ln)
        if not m_code:
            continue

        code = m_code.group("code")
        monies = list(money_iter_re.finditer(ln))
        if len(monies) < 6:
            continue

        vals = [Decimal(m.group(0).replace(",", "")) for m in monies]
        gp_match = pct_re.search(ln)
        gp_pct = Decimal(gp_match.group(1).replace(",", "")) if gp_match else None

        # expected order:
        # ext_price, ext_cost, profit, gp%, ext_tax, ext_price2, ext_price_tax, pct_total
        ext_price = vals[0]
        ext_cost = vals[1]
        profit = vals[2]
        ext_tax = vals[4]
        sales_plus_tax = vals[6] if len(vals) >= 7 else None
        pct_total = vals[7] if len(vals) >= 8 else None

        # class name between ext_tax and ext_price2
        name = ""
        try:
            tax_match = monies[4]
            price2_match = monies[5]
            name = ln[tax_match.end():price2_match.start()].strip()
            name = re.sub(r"\s+", " ", name)
        except Exception:
            name = ""

        if not name:
            name = "UNKNOWN"

        out.append(
            {
                "class_code": code,
                "class_name": name,
                "sales": ext_price,
                "tax": ext_tax,
                "sales_plus_tax": sales_plus_tax,
                "cost": ext_cost,
                "profit": profit,
                "gp_pct": gp_pct,
                "pct_total": pct_total,
            }
        )

    if not out:
        sample = "\n".join(lines[:120])
        raise ValueError(f"Could not extract any salesclass rows. Sample:\n{sample}")

    return out


# -----------------------------------------------------------------------------
# PayType parsing (Daily Invoice List / ARInvRegS)
# -----------------------------------------------------------------------------

def extract_paytype_totals(pdf_bytes: bytes) -> list[dict]:
    """
    Parse the *summary totals* by payment type.

    In the PDF, each payment-type section ends with a line that includes:
      invoice_total, tax_amount, (PaymentType) Payment Type: taxed_sales nontaxed_sales

    Sometimes TransAct splits that across lines like:
      1,504.2490.81CC
      Payment Type:
      329.10 1,084.33

    We capture:
      invoice_total (gross), tax_amount, taxed_sales, nontaxed_sales
      sales_total = taxed_sales + nontaxed_sales
      amount = invoice_total  (keeps compatibility with what you already had working)
      txn_count = count of invoice lines under that payment type (best-effort)
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]

    header_re = re.compile(r"^(?P<ptype>.+?)Payment Type:$")
    invoice_line_re = re.compile(r"^\d{2}/\d{2}/\d{2}\s")

    inline_totals_re = re.compile(
        rf"(?P<invoice_total>{_money})\s*(?P<tax>{_money})\s*(?P<ptype>.+?)Payment Type:\s*(?P<taxed>{_money})\s*(?P<nontaxed>{_money})$"
    )
    split_start_re = re.compile(rf"^(?P<invoice_total>{_money})\s*(?P<tax>{_money})\s*(?P<ptype>[A-Za-z/]+)$")
    two_money_re = re.compile(rf"^(?P<taxed>{_money})\s*(?P<nontaxed>{_money})$")

    def norm_ptype(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip())

    txn_counts = defaultdict(int)
    current_ptype = None

    totals: dict[str, dict] = {}
    pending = None
    waiting_for_two_money = False

    for ln in lines:
        hm = header_re.match(ln)
        if hm:
            current_ptype = norm_ptype(hm.group("ptype"))
            continue

        if invoice_line_re.match(ln) and current_ptype:
            txn_counts[current_ptype] += 1

        m = inline_totals_re.match(ln)
        if m:
            p = norm_ptype(m.group("ptype"))
            invoice_total = _to_decimal(m.group("invoice_total"))
            tax_amount = _to_decimal(m.group("tax"))
            taxed_sales = _to_decimal(m.group("taxed"))
            nontaxed_sales = _to_decimal(m.group("nontaxed"))

            if invoice_total is None or tax_amount is None or taxed_sales is None or nontaxed_sales is None:
                continue

            totals[p] = {
                "payment_type": p,
                "amount": invoice_total,  # legacy column
                "invoice_total": invoice_total,
                "tax_amount": tax_amount,
                "taxed_sales": taxed_sales,
                "nontaxed_sales": nontaxed_sales,
                "sales_total": taxed_sales + nontaxed_sales,
                "txn_count": txn_counts.get(p),
            }
            continue

        m = split_start_re.match(ln)
        if m:
            pending = {
                "ptype": norm_ptype(m.group("ptype")),
                "invoice_total": _to_decimal(m.group("invoice_total")),
                "tax_amount": _to_decimal(m.group("tax")),
            }
            waiting_for_two_money = False
            continue

        if ln == "Payment Type:" and pending:
            waiting_for_two_money = True
            continue

        if waiting_for_two_money and pending:
            m2 = two_money_re.match(ln)
            if m2:
                p = pending["ptype"]
                taxed_sales = _to_decimal(m2.group("taxed"))
                nontaxed_sales = _to_decimal(m2.group("nontaxed"))
                invoice_total = pending["invoice_total"]
                tax_amount = pending["tax_amount"]

                if invoice_total is None or tax_amount is None or taxed_sales is None or nontaxed_sales is None:
                    pending = None
                    waiting_for_two_money = False
                    continue

                totals[p] = {
                    "payment_type": p,
                    "amount": invoice_total,  # legacy column
                    "invoice_total": invoice_total,
                    "tax_amount": tax_amount,
                    "taxed_sales": taxed_sales,
                    "nontaxed_sales": nontaxed_sales,
                    "sales_total": taxed_sales + nontaxed_sales,
                    "txn_count": txn_counts.get(p),
                }
                pending = None
                waiting_for_two_money = False
                continue

    if not totals:
        sample = "\n".join(lines[:160])
        raise ValueError(f"Could not extract any payment type totals. Sample:\n{sample}")

    return list(totals.values())


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


# -------------------- SalesStat endpoints --------------------

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
                        {"ok": False, "error": "Latest salesstat PDF missing business_date", "id": file_id}
                    ), 400

                kpis = extract_salesstat_kpis(pdf_bytes)

                # upsert row (don’t break existing)
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

                file_id, _, filename, sha256, pdf_bytes = row
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


# -------------------- SalesClass endpoints --------------------

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
                    return jsonify({"ok": False, "error": "Latest salesclass PDF missing business_date", "id": file_id}), 400

                rows = extract_salesclass_rows(pdf_bytes)

                # refresh day (delete+insert) so we don't rely on knowing your unique constraint
                cur.execute("delete from analytics.salesclass_daily where business_date = %s", (biz_date,))

                # attach provenance if columns exist
                for r in rows:
                    r["business_date"] = biz_date
                    r["source_file_id"] = file_id
                    r["source_sha256"] = sha256

                inserted = _insert_rows(cur, "analytics", "salesclass_daily", rows)

        return jsonify(
            {
                "ok": True,
                "source": {"id": file_id, "filename": filename, "sha256": sha256, "business_date": str(biz_date)},
                "inserted_rows": inserted,
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
                    where report_type = 'salesclass'
                      and business_date = %s
                    order by received_at desc
                    limit 1
                    """,
                    (biz_date,),
                )
                row = cur.fetchone()
                if not row:
                    return jsonify({"ok": False, "error": f"no salesclass PDF found for {business_date_str}"}), 404

                file_id, _, filename, sha256, pdf_bytes = row
                rows = extract_salesclass_rows(pdf_bytes)

                cur.execute("delete from analytics.salesclass_daily where business_date = %s", (biz_date,))

                for r in rows:
                    r["business_date"] = biz_date
                    r["source_file_id"] = file_id
                    r["source_sha256"] = sha256

                inserted = _insert_rows(cur, "analytics", "salesclass_daily", rows)

        return jsonify(
            {
                "ok": True,
                "source": {"id": file_id, "filename": filename, "sha256": sha256, "business_date": str(biz_date)},
                "inserted_rows": inserted,
            }
        )

    except Exception as e:
        app.logger.exception("parse_salesclass_by_date failed")
        return jsonify({"ok": False, "error_type": type(e).__name__, "error": str(e)}), 500


# -------------------- PayType endpoints --------------------

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
                    return jsonify({"ok": False, "error": "Latest paytype PDF missing business_date", "id": file_id}), 400

                rows = extract_paytype_totals(pdf_bytes)

                # refresh day (delete+insert)
                cur.execute("delete from analytics.paytype_daily where business_date = %s", (biz_date,))

                for r in rows:
                    r["business_date"] = biz_date
                    r["source_file_id"] = file_id
                    r["source_sha256"] = sha256

                inserted = _insert_rows(cur, "analytics", "paytype_daily", rows)

        return jsonify(
            {
                "ok": True,
                "source": {"id": file_id, "filename": filename, "sha256": sha256, "business_date": str(biz_date)},
                "inserted_rows": inserted,
            }
        )

    except Exception as e:
        app.logger.exception("parse_paytype_latest failed")
        return jsonify({"ok": False, "error_type": type(e).__name__, "error": str(e)}), 500


@app.post("/parse/paytype/by-date")
def parse_paytype_by_date():
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
                    where report_type = 'paytype'
                      and business_date = %s
                    order by received_at desc
                    limit 1
                    """,
                    (biz_date,),
                )
                row = cur.fetchone()
                if not row:
                    return jsonify({"ok": False, "error": f"no paytype PDF found for {business_date_str}"}), 404

                file_id, _, filename, sha256, pdf_bytes = row
                rows = extract_paytype_totals(pdf_bytes)

                cur.execute("delete from analytics.paytype_daily where business_date = %s", (biz_date,))

                for r in rows:
                    r["business_date"] = biz_date
                    r["source_file_id"] = file_id
                    r["source_sha256"] = sha256

                inserted = _insert_rows(cur, "analytics", "paytype_daily", rows)

        return jsonify(
            {
                "ok": True,
                "source": {"id": file_id, "filename": filename, "sha256": sha256, "business_date": str(biz_date)},
                "inserted_rows": inserted,
            }
        )

    except Exception as e:
        app.logger.exception("parse_paytype_by_date failed")
        return jsonify({"ok": False, "error_type": type(e).__name__, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
