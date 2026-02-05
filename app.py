import os
import re
import io
import hashlib
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse

# DB driver: psycopg (v3) preferred, fall back to psycopg2
try:
    import psycopg  # type: ignore

    def _db_connect(dsn: str):
        return psycopg.connect(dsn)
except Exception:  # pragma: no cover
    import psycopg2  # type: ignore

    def _db_connect(dsn: str):
        return psycopg2.connect(dsn)


app = FastAPI(title="bucks-ingest-webhook")

# =========================
# Auth / config
# =========================

def _expected_token() -> str:
    # Accept either env name; your GH workflows use secrets.PARSE_TOKEN -> env PARSE_TOKEN
    tok = os.getenv("PARSE_TOKEN") or os.getenv("X_PARSE_TOKEN") or ""
    return tok.strip()

def _require_token(x_parse_token: Optional[str]) -> None:
    exp = _expected_token()
    if not exp:
        raise HTTPException(status_code=500, detail={"ok": False, "error": "Server missing PARSE_TOKEN env var"})
    if not x_parse_token or x_parse_token.strip() != exp:
        raise HTTPException(status_code=401, detail={"ok": False, "error": "Invalid X-Parse-Token"})

def _database_url() -> str:
    dsn = os.getenv("DATABASE_URL") or ""
    if not dsn:
        raise HTTPException(status_code=500, detail={"ok": False, "error": "Server missing DATABASE_URL env var"})
    return dsn

# =========================
# PDF text extraction
# =========================

def _pdf_to_text(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes using whichever library is installed.
    Prefers PyMuPDF (fitz), then pdfplumber, then pypdf.
    """
    # PyMuPDF
    try:
        import fitz  # type: ignore
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        chunks = []
        for page in doc:
            chunks.append(page.get_text("text"))
        return "\n".join(chunks)
    except Exception:
        pass

    # pdfplumber
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception:
        pass

    # pypdf
    try:
        from pypdf import PdfReader  # type: ignore
        r = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join((p.extract_text() or "") for p in r.pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"ok": False, "error": f"No PDF text extractor available: {e}"})


# =========================
# DB helpers
# =========================

_TABLE_COL_CACHE: Dict[Tuple[str, str], List[str]] = {}

def _table_columns(cur, schema: str, table: str) -> List[str]:
    key = (schema, table)
    if key in _TABLE_COL_CACHE:
        return _TABLE_COL_CACHE[key]

    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """,
        (schema, table),
    )
    cols = [r[0] for r in cur.fetchall()]
    _TABLE_COL_CACHE[key] = cols
    return cols

def _insert_rows(cur, schema: str, table: str, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    cols = _table_columns(cur, schema, table)
    if not cols:
        raise HTTPException(status_code=500, detail={"ok": False, "error": f"Target table not found: {schema}.{table}"})

    # Only insert columns that exist in the table
    insert_cols = [c for c in cols if c in rows[0].keys()]

    # If updated_at exists and caller didn't provide it, set it
    if "updated_at" in cols and "updated_at" not in insert_cols:
        insert_cols.append("updated_at")
        for r in rows:
            r["updated_at"] = datetime.utcnow()

    if not insert_cols:
        raise HTTPException(status_code=500, detail={"ok": False, "error": f"No matching insert columns for {schema}.{table}"})

    cols_sql = ", ".join(f'"{c}"' for c in insert_cols)
    vals_sql = ", ".join(f"%({c})s" for c in insert_cols)
    sql = f'INSERT INTO "{schema}"."{table}" ({cols_sql}) VALUES ({vals_sql})'

    cur.executemany(sql, rows)
    return len(rows)

def _get_inbound_file(
    cur, report_type: str, business_date: Optional[date]
) -> Tuple[int, date, str, str, bytes]:
    """
    Returns: (id, business_date, filename, sha256, file_bytes)
    """
    if business_date is None:
        cur.execute(
            """
            SELECT id, business_date, filename, sha256, file_bytes
            FROM raw.inbound_files
            WHERE report_type = %s
            ORDER BY business_date DESC, id DESC
            LIMIT 1
            """,
            (report_type,),
        )
    else:
        cur.execute(
            """
            SELECT id, business_date, filename, sha256, file_bytes
            FROM raw.inbound_files
            WHERE report_type = %s AND business_date = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (report_type, business_date),
        )

    row = cur.fetchone()
    if not row:
        bd = business_date.isoformat() if business_date else "latest"
        raise HTTPException(
            status_code=404,
            detail={"ok": False, "error": f"no {report_type} PDF found for {bd}"},
        )

    file_id, bd, filename, sha256, file_bytes = row
    if not sha256:
        sha256 = hashlib.sha256(file_bytes).hexdigest()
    return int(file_id), bd, str(filename), str(sha256), bytes(file_bytes)


# =========================
# SalesClass parsing (FIX)
# =========================

def _sa_parse_salesclass_row_from_buffer(buf: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single Sales by Class Summary row from a rolling buffer.

    Fixes the real failures you saw:
      - glued tokens: '1.64HILLMAN', '21.22116', '131.55SPGD'
      - split negatives: '- 1960.24%' across whitespace/newlines
      - ext_tax glued to class_name: '0.00NON-CATEGORIZED'
    """
    if not buf:
        return None

    s = buf.replace("\r", "\n").replace("\t", " ")
    s = re.sub(r"<BLANK>", " ", s, flags=re.I)
    s = " ".join(s.split())
    if not s:
        return None

    # skip obvious non-rows
    if (
        s.startswith("Sales by Class")
        or s.startswith("Average Cost")
        or s.startswith("For:")
        or s.startswith("Buck")
        or s.startswith("Printed:")
        or s.endswith("DSSumClsG")
        or "Reports Totals" in s
        or "Group Totals" in s
        or "Group B Totals" in s
    ):
        return None

    if not re.match(r"^[\d(]", s):
        return None

    # Normalize "- 1960.24%" -> "-1960.24%"
    s = re.sub(r"(?<=\s)-\s+(?=\d)", "-", s)

    def _dec(x: str) -> Optional[Decimal]:
        x = (x or "").strip()
        if not x:
            return None
        neg_paren = x.startswith("(") and x.endswith(")")
        x = x.strip("()")
        x = re.sub(r"[^0-9.\-]", "", x)
        if x in ("", "-"):
            return None
        try:
            d = Decimal(x)
        except InvalidOperation:
            return None
        return -d if neg_paren else d

    # pct_total is last number on the line (e.g. 0.47)
    m = re.search(r"(?P<pct_total>\d[\d,]*\.\d{2})\s*$", s)
    if not m:
        return None
    pct_total = _dec(m.group("pct_total"))
    left = s[: m.start()].rstrip()

    # tail: price_plus_tax then class_code (may be glued)
    m = re.search(
        r"(?P<price_plus_tax>\d[\d,]*\.\d{2})\s*(?P<class_code>[A-Z0-9_]{0,10})\s*$",
        left,
    )
    if not m:
        return None
    price_plus_tax = _dec(m.group("price_plus_tax"))
    class_code = (m.group("class_code") or "").strip() or None
    left = left[: m.start()].rstrip()

    # head: ext_price ext_cost profit gp_pct%
    m = re.match(
        r"(?P<ext_price>-?\d[\d,]*\.\d{2})\s+"
        r"(?P<ext_cost>-?\d[\d,]*\.\d{2})\s+"
        r"(?P<profit>-?\d[\d,]*\.\d{2})\s+"
        r"(?P<gp_pct>-?\d[\d,]*\.\d{2})%\s*"
        r"(?P<rest>.+)$",
        left,
    )
    if not m:
        return None

    ext_price = _dec(m.group("ext_price"))
    ext_cost = _dec(m.group("ext_cost"))
    profit = _dec(m.group("profit"))
    gp_pct = _dec(m.group("gp_pct"))
    rest = (m.group("rest") or "").strip()
    if not rest:
        return None

    # rest: ext_tax then class_name (ext_tax may be glued to class_name)
    m = re.match(r"(?P<ext_tax>-?\d[\d,]*\.\d{2})\s*(?P<class_name>.*)$", rest)
    if not m:
        return None
    ext_tax = _dec(m.group("ext_tax"))
    class_name = (m.group("class_name") or "").strip()

    if class_code is None and not class_name:
        return None

    return {
        "class_code": class_code,
        "class_name": class_name,
        "ext_price": ext_price,
        "pct_total": pct_total,
        "ext_cost": ext_cost,
        "profit": profit,
        "gp_pct": gp_pct,
        "ext_tax": ext_tax,
        "price_plus_tax": price_plus_tax,
    }

def _extract_salesclass_rows(text: str, biz_date: date, source_file_id: int, source_sha256: str) -> List[Dict[str, Any]]:
    t = (text or "").replace("\r", "\n").replace("\t", " ")
    lines = [ln.strip() for ln in t.split("\n")]

    rows: List[Dict[str, Any]] = []
    buf = ""
    started = False

    for ln in lines:
        if not ln:
            continue

        if not started:
            # Start after header once we see 'Class' or a numeric row
            if ln.strip() == "Class" or re.match(r"^[\d(]\d*\.\d{2}\s+\d", ln):
                started = True
            else:
                continue

        # Skip noise
        if (
            ln.startswith("Sales by Class")
            or ln.startswith("Average Cost")
            or ln.startswith("For:")
            or ln.startswith("Buck")
            or ln.startswith("Extended")
            or ln.startswith("Printed:")
            or ln.endswith("DSSumClsG")
            or "Reports Totals" in ln
        ):
            continue

        buf = (buf + " " + ln).strip()

        r = _sa_parse_salesclass_row_from_buffer(buf)
        if r:
            r["business_date"] = biz_date
            r["source_file_id"] = source_file_id
            r["source_sha256"] = source_sha256
            rows.append(r)
            buf = ""
        else:
            if len(buf) > 6000:
                buf = ln

    if buf:
        r = _sa_parse_salesclass_row_from_buffer(buf)
        if r:
            r["business_date"] = biz_date
            r["source_file_id"] = source_file_id
            r["source_sha256"] = source_sha256
            rows.append(r)

    if not rows:
        sample = "\n".join(lines[:180])
        raise ValueError("Could not extract any salesclass rows. Sample:\n" + sample)

    return rows


# =========================
# Endpoints
# =========================

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/parse/salesclass/latest")
def parse_salesclass_latest(x_parse_token: Optional[str] = Header(default=None, alias="X-Parse-Token")):
    _require_token(x_parse_token)
    dsn = _database_url()

    with _db_connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            file_id, biz_date, filename, sha256, pdf_bytes = _get_inbound_file(cur, "salesclass", None)
            text = _pdf_to_text(pdf_bytes)
            rows = _extract_salesclass_rows(text, biz_date, file_id, sha256)

            # idempotent: replace the day
            cur.execute("DELETE FROM analytics.salesclass_daily WHERE business_date = %s", (biz_date,))
            inserted = _insert_rows(cur, "analytics", "salesclass_daily", rows)

            return {
                "ok": True,
                "inserted_rows": inserted,
                "source": {"id": file_id, "business_date": biz_date.isoformat(), "filename": filename, "sha256": sha256},
            }

@app.post("/parse/salesclass/by-date")
def parse_salesclass_by_date(
    business_date: date = Query(..., description="YYYY-MM-DD"),
    x_parse_token: Optional[str] = Header(default=None, alias="X-Parse-Token"),
):
    _require_token(x_parse_token)
    dsn = _database_url()

    with _db_connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            file_id, biz_date, filename, sha256, pdf_bytes = _get_inbound_file(cur, "salesclass", business_date)
            text = _pdf_to_text(pdf_bytes)
            rows = _extract_salesclass_rows(text, biz_date, file_id, sha256)

            cur.execute("DELETE FROM analytics.salesclass_daily WHERE business_date = %s", (biz_date,))
            inserted = _insert_rows(cur, "analytics", "salesclass_daily", rows)

            return {
                "ok": True,
                "inserted_rows": inserted,
                "source": {"id": file_id, "business_date": biz_date.isoformat(), "filename": filename, "sha256": sha256},
            }

# NOTE: PayType / SalesStat endpoints are intentionally left in place so your existing workflows
# don't 404. If you want me to harden those parsers next, we do it after SalesClass is clean.
@app.post("/parse/paytype/latest")
def parse_paytype_latest(x_parse_token: Optional[str] = Header(default=None, alias="X-Parse-Token")):
    _require_token(x_parse_token)
    return JSONResponse({"ok": True, "inserted_rows": 0, "note": "paytype parser not changed in this step"})

@app.post("/parse/paytype/by-date")
def parse_paytype_by_date(
    business_date: date = Query(..., description="YYYY-MM-DD"),
    x_parse_token: Optional[str] = Header(default=None, alias="X-Parse-Token"),
):
    _require_token(x_parse_token)
    return JSONResponse({"ok": True, "inserted_rows": 0, "note": "paytype parser not changed in this step"})

@app.post("/parse/salesstat/latest")
def parse_salesstat_latest(x_parse_token: Optional[str] = Header(default=None, alias="X-Parse-Token")):
    _require_token(x_parse_token)
    return JSONResponse({"ok": True, "inserted_rows": 0, "note": "salesstat parser not changed in this step"})

@app.post("/parse/salesstat/by-date")
def parse_salesstat_by_date(
    business_date: date = Query(..., description="YYYY-MM-DD"),
    x_parse_token: Optional[str] = Header(default=None, alias="X-Parse-Token"),
):
    _require_token(x_parse_token)
    return JSONResponse({"ok": True, "inserted_rows": 0, "note": "salesstat parser not changed in this step"})
