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
    # Invoice List Summary (payment type subtotals live here)
    "arinvregreport": "paytype",
}


def infer_report_type(filename: str | None, subject: str | None) -> str | None:
    fn = (filename or "").lower()
    sub = (subject or "").lower()

    for prefix, rtype in _REPORT_PREFIX_MAP.items():
        if fn.startswith(prefix):
            return rtype

    if "daily sales statistics" in sub:
        return "salesstat"
    if "sales by class" in sub:
        return "salesclass"
    if "invoice list" in sub:
        return "paytype"

    return None


def infer_business_date(filename: str | None, subject: str | None) -> date | None:
    """
    Try to infer business date from common patterns.
    Examples:
      salesstatreport012526020108.PDF  -> MMDDYY = 01/25/26
      salesclassreport012726010024.PDF -> MMDDYY = 01/27/26
      arinvregreport012726010235.PDF   -> MMDDYY = 01/27/26
    """
    fn = filename or ""

    m = re.search(r"(salesstatreport|salesclassreport|arinvregreport)(\d{6})", fn, flags=re.IGNORECASE)
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


def _pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join([(p.extract_text() or "") for p in reader.pages])


def _norm_lines(text: str) -> list[str]:
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln]


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
    text = _pdf_text(pdf_bytes)
    lines = _norm_lines(text)

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


# -----------------------
# SalesClass extractor
# -----------------------
_salesclass_num_re = re.compile(r"-?\d[\d,]*\.\d{2}%?")


def extract_salesclass_rows(pdf_bytes: bytes) -> list[dict]:
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
        if re.match(r"^\d[\d,]*\.\d{2}$", class_code):
            continue

        matches = list(_salesclass_num_re.finditer(l))
        if len(matches) < 5:
            continue

        ext_price = matches[0].group()
        ext_cost = matches[1].group()
        profit = matches[2].group()
        gp_pct = matches[3].group().replace("%", "")
        ext_tax = matches[4].group()

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


# -----------------------
# PayType extractor (Invoice List Summary / ARInvRegS)
# -----------------------
def extract_paytype_rows(pdf_bytes: bytes) -> list[dict]:
    """
    This report DOES NOT have a nice "tender summary table".
    It has payment-type subtotals in lines like:

      1,107.2550.86Acct ChgPayment Type: 449.10 607.29

    Interpretation:
      - First number: total SALES for that payment type (pre-tax)
      - Second number: total TAX for that payment type
      - After 'Payment Type:' the order is:
          Non-Taxed sales FIRST, Taxed sales SECOND  ✅ (confirmed by tax-rate math)

    For now, we store:
      amount = total SALES (pre-tax)
      txn_count = NULL (we can add counts later if you want)
    """
    text = _pdf_text(pdf_bytes)
    lines = _norm_lines(text)

    money_pat = re.compile(r"-?\d[\d,]*\.\d{2}")

    out: list[dict] = []

    i = 0
    while i < len(lines):
        ln = lines[i]

        # Case A: one-line subtotal contains 'Payment Type:'
        if "Payment Type:" in ln:
            left, right = ln.split("Payment Type:", 1)

            left_m = money_pat.findall(left)
            right_m = money_pat.findall(right)

            # We need: [sales, tax] on left and [nontaxed, taxed] on right
            if len(left_m) >= 2 and len(right_m) >= 2:
                sales = _money_to_decimal(left_m[0])
                tax = _money_to_decimal(left_m[1])

                # label: remove the first 2 money tokens from left
                tmp = left
                tmp = tmp.replace(left_m[0], "", 1).replace(left_m[1], "", 1)
                label = re.sub(r"\s+", " ", tmp).strip()

                # right side order confirmed: Non-Taxed then Taxed
                nontaxed = _money_to_decimal(right_m[0])
                taxed = _money_to_decimal(right_m[1])

                if label and sales is not None:
                    out.append(
                        {
                            "payment_type": label,
                            "amount": sales,              # what we chart (sales by payment type)
                            "txn_count": None,
                            # debug fields (not stored in DB today)
                            "_tax": tax,
                            "_nontaxed_sales": nontaxed,
                            "_taxed_sales": taxed,
                        }
                    )
            i += 1
            continue

        # Case B: three-line format:
        #   line i:   "<sales><tax><label>"  (no 'Payment Type:')
        #   line i+1: "Payment Type:"
        #   line i+2: "<nontaxed> <taxed>"
        if i + 2 < len(lines) and lines[i + 1].strip() == "Payment Type:":
            first = lines[i]
            third = lines[i + 2]
            m1 = money_pat.findall(first)
            m3 = money_pat.findall(third)

            if len(m1) >= 2 and len(m3) >= 2:
                sales = _money_to_decimal(m1[0])
                tax = _money_to_decimal(m1[1])

                tmp = first
                tmp = tmp.replace(m1[0], "", 1).replace(m1[1], "", 1)
                label = re.sub(r"\s+", " ", tmp).strip()

                # third line order confirmed: Non-Taxed then Taxed
                nontaxed = _money_to_decimal(m3[0])
                taxed = _money_to_decimal(m3[1])

                if label and sales is not None:
                    out.append(
                        {
                            "payment_type": label,
                            "amount": sales,
                            "txn_count": None,
                            "_tax": tax,
                            "_nontaxed_sales": nontaxed,
                            "_taxed_sales": taxed,
                        }
                    )
                    i += 3
                    continue

        i += 1

    if not out:
        sample = "\n".join(lines[:140])
        raise ValueError(f"Could not extract any payment type subtotals. Sample:\n{sample}")

    return out


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
                        {"ok": False, "error": "Latest salesstat PDF missing business_date; cannot parse"}, 400
                    )

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
        business_date = datetime.strptime(business_date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"ok": False, "error": "invalid business_date (expected YYYY-MM-DD)"}), 400

    try:
        with connect(get_db_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, filename, sha256, file_bytes, business_date
                    from raw.inbound_files
                    where report_type = 'salesstat'
                      and business_date = %s
                    order by received_at desc
                    limit 1
                    """,
                    (business_date,),
                )
                row = cur.fetchone()

        if not row:
            return jsonify({"ok": False, "error": f"no salesstat PDF found for {business_date_str}"}), 404

        file_id, filename, sha256, pdf_bytes, biz_date = row
        kpis = extract_salesstat_kpis(pdf_bytes)

        with connect(get_db_url()) as conn:
            with conn.cursor() as cur:
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

                # upsert all rows
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
                            file_id,
                            sha256,
                        ),
                    )
                    n += 1

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
        business_date = datetime.strptime(business_date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"ok": False, "error": "invalid business_date (expected YYYY-MM-DD)"}), 400

    try:
        with connect(get_db_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, business_date, filename, sha256, file_bytes
                    from raw.inbound_files
                    where report_type='salesclass'
                      and business_date=%s
                    order by received_at desc
                    limit 1
                    """,
                    (business_date,),
                )
                row = cur.fetchone()
                if not row:
                    return jsonify({"ok": False, "error": f"No salesclass PDF found for {business_date_str}"}), 404

                file_id, biz_date, filename, sha256, pdf_bytes = row
                rows = extract_salesclass_rows(pdf_bytes)

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
                            file_id,
                            sha256,
                        ),
                    )
                    n += 1

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


# --- PayType endpoints ---
@app.post("/parse/paytype/latest")
def parse_paytype_latest():
    try:
        require_parse_token()
    except PermissionError as e:
        return jsonify({"ok": False, "error": str(e)}), 401

    try:
        with connect(get_db_url()) as conn:
            with conn.cursor() as cur:
                # catch older rows that didn't infer report_type
                cur.execute(
                    """
                    select id, business_date, filename, mail_subject, sha256, file_bytes
                    from raw.inbound_files
                    where (report_type='paytype' or filename ilike 'arinvregreport%')
                    order by received_at desc
                    limit 1
                    """
                )
                row = cur.fetchone()
                if not row:
                    return jsonify({"ok": False, "error": "No paytype PDFs found in raw.inbound_files"}), 404

                file_id, biz_date, filename, mail_subject, sha256, pdf_bytes = row

                # backfill report_type and business_date if missing
                if not biz_date:
                    biz_date = infer_business_date(filename, mail_subject)
                    if biz_date:
                        cur.execute("update raw.inbound_files set business_date=%s where id=%s", (biz_date, file_id))
                    else:
                        return jsonify({"ok": False, "error": "Paytype PDF missing business_date"}), 400

                cur.execute("update raw.inbound_files set report_type='paytype' where id=%s", (file_id,))

                rows = extract_paytype_rows(pdf_bytes)

                # upsert into analytics.paytype_daily (current schema)
                n = 0
                for r in rows:
                    cur.execute(
                        """
                        insert into analytics.paytype_daily
                          (business_date, payment_type, amount, txn_count, source_file_id, source_sha256, updated_at)
                        values
                          (%s, %s, %s, %s, %s, %s, now())
                        on conflict (business_date, payment_type) do update
                          set amount = excluded.amount,
                              txn_count = excluded.txn_count,
                              source_file_id = excluded.source_file_id,
                              source_sha256 = excluded.source_sha256,
                              updated_at = now()
                        """,
                        (
                            biz_date,
                            r["payment_type"],
                            r["amount"],
                            r["txn_count"],
                            file_id,
                            sha256,
                        ),
                    )
                    n += 1

        return jsonify(
            {
                "ok": True,
                "source": {"id": file_id, "filename": filename, "sha256": sha256, "business_date": str(biz_date)},
                "rows_upserted": n,
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
        target_date = datetime.strptime(business_date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"ok": False, "error": "invalid business_date (expected YYYY-MM-DD)"}), 400

    mmddyy = target_date.strftime("%m%d%y")

    try:
        with connect(get_db_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, business_date, filename, mail_subject, sha256, file_bytes
                    from raw.inbound_files
                    where
                      (report_type='paytype' and business_date=%s)
                      or (filename ilike 'arinvregreport%' and (
                            business_date=%s
                            or (business_date is null and filename ilike %s)
                         ))
                    order by received_at desc
                    limit 1
                    """,
                    (target_date, target_date, f"%{mmddyy}%"),
                )
                row = cur.fetchone()

                if not row:
                    return jsonify({"ok": False, "error": f"No paytype PDF found for {business_date_str}"}), 404

                file_id, biz_date, filename, mail_subject, sha256, pdf_bytes = row

                if not biz_date:
                    biz_date = infer_business_date(filename, mail_subject) or target_date
                    cur.execute("update raw.inbound_files set business_date=%s where id=%s", (biz_date, file_id))

                cur.execute("update raw.inbound_files set report_type='paytype' where id=%s", (file_id,))

                rows = extract_paytype_rows(pdf_bytes)

                n = 0
                for r in rows:
                    cur.execute(
                        """
                        insert into analytics.paytype_daily
                          (business_date, payment_type, amount, txn_count, source_file_id, source_sha256, updated_at)
                        values
                          (%s, %s, %s, %s, %s, %s, now())
                        on conflict (business_date, payment_type) do update
                          set amount = excluded.amount,
                              txn_count = excluded.txn_count,
                              source_file_id = excluded.source_file_id,
                              source_sha256 = excluded.source_sha256,
                              updated_at = now()
                        """,
                        (
                            biz_date,
                            r["payment_type"],
                            r["amount"],
                            r["txn_count"],
                            file_id,
                            sha256,
                        ),
                    )
                    n += 1

        return jsonify(
            {
                "ok": True,
                "source": {"id": file_id, "filename": filename, "sha256": sha256, "business_date": str(biz_date)},
                "rows_upserted": n,
            }
        )

    except Exception as e:
        app.logger.exception("parse_paytype_by_date failed")
        return jsonify({"ok": False, "error_type": type(e).__name__, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
