import os
import re
import io
import json
import time
import uuid
import hashlib
import logging
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException as FastAPIHTTPException

# DB driver: psycopg (v3) preferred, fall back to psycopg2
try:
    import psycopg  # type: ignore

    def _db_connect(dsn: str):
        return psycopg.connect(dsn)

except Exception:  # pragma: no cover
    import psycopg2  # type: ignore

    def _db_connect(dsn: str):
        return psycopg2.connect(dsn)


# -----------------------------------------------------------------------------
# Logging (JSON lines)
# -----------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(message)s")
logger = logging.getLogger("bucks-ingest")


def _log(event: str, **fields: Any) -> None:
    payload = {"ts": datetime.utcnow().isoformat() + "Z", "event": event, **fields}
    try:
        logger.info(json.dumps(payload, default=str))
    except Exception:
        logger.info(f"{event} | {fields}")


# -----------------------------------------------------------------------------
# App + safety: never crash worker; structured JSON errors; avoid 500s
# -----------------------------------------------------------------------------

app = FastAPI(title="bucks-ingest-webhook")


def _json_error(
    request_id: str,
    error_type: str,
    message: str,
    *,
    http_status: int = 200,
    extra: Optional[Dict[str, Any]] = None,
):
    payload: Dict[str, Any] = {
        "ok": False,
        "request_id": request_id,
        "error_type": error_type,
        "message": message,
    }
    if extra:
        payload.update(extra)
    return JSONResponse(status_code=http_status, content=payload)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
    start = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        # Last-resort catch: never crash the worker, never emit 500 stack traces to client.
        _log(
            "request.unhandled_exception",
            request_id=request_id,
            path=str(request.url.path),
            query=str(request.url.query),
            method=request.method,
            error=str(e),
        )
        response = _json_error(request_id, "exception", str(e), http_status=200)

    # attach request id + timing
    try:
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Elapsed-Ms"] = str(int((time.time() - start) * 1000))
    except Exception:
        pass
    return response


@app.exception_handler(FastAPIHTTPException)
async def http_exception_handler(request: Request, exc: FastAPIHTTPException):
    request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
    status = int(getattr(exc, "status_code", 200) or 200)

    # Keep real 4xx for client errors; downgrade any 5xx to 200 with structured JSON.
    http_status = status if status < 500 else 200

    detail = getattr(exc, "detail", None)
    extra: Optional[Dict[str, Any]] = None
    if isinstance(detail, dict):
        message = detail.get("error") or detail.get("message") or str(detail)
        extra = {k: v for k, v in detail.items() if k not in ("error", "message")}
    else:
        message = str(detail) if detail else str(exc)

    # default classification by status
    et = "http_error"
    if status == 401:
        et = "unauthorized"
    elif status == 404:
        et = "not_found"
    elif status == 409:
        et = "conflict"

    # allow explicit error_type override from detail
    if extra and isinstance(extra.get("error_type"), str):
        et = str(extra["error_type"])

    _log(
        "request.http_exception",
        request_id=request_id,
        path=str(request.url.path),
        query=str(request.url.query),
        method=request.method,
        status=status,
        error_type=et,
        message=message,
    )
    return _json_error(request_id, et, message, http_status=http_status, extra=extra)


# -----------------------------------------------------------------------------
# Auth / config
# -----------------------------------------------------------------------------

def _expected_token() -> str:
    # Accept either env name; GitHub uses secrets.PARSE_TOKEN -> env PARSE_TOKEN
    tok = os.getenv("PARSE_TOKEN") or os.getenv("X_PARSE_TOKEN") or ""
    return tok.strip()


def _require_token(x_parse_token: Optional[str]) -> None:
    exp = _expected_token()
    if not exp:
        raise FastAPIHTTPException(
            status_code=200,
            detail={"error": "Server missing PARSE_TOKEN env var", "error_type": "server_misconfig"},
        )
    if not x_parse_token or x_parse_token.strip() != exp:
        # keep it 200 with a structured body (so curl/automation doesn't treat as hard failure)
        raise FastAPIHTTPException(
            status_code=200,
            detail={"error": "Invalid X-Parse-Token", "error_type": "unauthorized"},
        )


def _database_url() -> str:
    dsn = os.getenv("DATABASE_URL") or ""
    if not dsn:
        raise FastAPIHTTPException(
            status_code=200,
            detail={"error": "Server missing DATABASE_URL env var", "error_type": "server_misconfig"},
        )
    return dsn


# -----------------------------------------------------------------------------
# PDF text extraction
# -----------------------------------------------------------------------------

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
        raise FastAPIHTTPException(status_code=200, detail={"error": f"No PDF text extractor available: {e}"})


# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------

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


def _prepare_insert(cur, schema: str, table: str, rows: List[Dict[str, Any]]):
    """
    Build INSERT SQL and insert column list for rows.
    Returns (sql, insert_cols) or (None, None) if table not found.
    """
    if not rows:
        return None, None

    cols = _table_columns(cur, schema, table)
    if not cols:
        return None, None

    # Only insert columns that exist in the table
    insert_cols = [c for c in cols if c in rows[0].keys()]

    # If updated_at exists and caller didn't provide it, set it
    if "updated_at" in cols and "updated_at" not in insert_cols:
        insert_cols.append("updated_at")
        for r in rows:
            r["updated_at"] = datetime.utcnow()

    if not insert_cols:
        return None, None

    cols_sql = ", ".join(f'"{c}"' for c in insert_cols)
    vals_sql = ", ".join(f"%({c})s" for c in insert_cols)
    sql = f'INSERT INTO "{schema}"."{table}" ({cols_sql}) VALUES ({vals_sql})'
    return sql, insert_cols


def _safe_insert_rows(
    cur,
    schema: str,
    table: str,
    rows: List[Dict[str, Any]],
    *,
    request_id: str,
    debug: bool,
    max_error_samples: int = 5,
) -> Dict[str, Any]:
    """
    Insert rows into a table without aborting the day.
    - Uses savepoints to skip rows that violate constraints.
    - Returns counts + small error samples (when debug=true).
    """
    out: Dict[str, Any] = {"schema": schema, "table": table, "inserted": 0, "skipped_db": 0}
    if not rows:
        return out

    sql, insert_cols = _prepare_insert(cur, schema, table, rows)
    if not sql or not insert_cols:
        out["note"] = "table_missing_or_no_columns"
        _log("db.insert.skip_table", request_id=request_id, schema=schema, table=table, note=out["note"])
        return out

    error_samples: List[Dict[str, Any]] = []

    for idx, r in enumerate(rows):
        cur.execute("SAVEPOINT sp_row")
        try:
            cur.execute(sql, r)
            out["inserted"] += 1
            cur.execute("RELEASE SAVEPOINT sp_row")
        except Exception as e:
            out["skipped_db"] += 1
            try:
                cur.execute("ROLLBACK TO SAVEPOINT sp_row")
                cur.execute("RELEASE SAVEPOINT sp_row")
            except Exception:
                pass

            if debug and len(error_samples) < max_error_samples:
                sample = {k: r.get(k) for k in ("class_code", "class_name", "ext_price", "pct_total")}
                sample["error"] = str(e)
                sample["row_index"] = idx
                error_samples.append(sample)

            _log(
                "db.insert.row_failed",
                request_id=request_id,
                schema=schema,
                table=table,
                row_index=idx,
                error=str(e),
            )

    if debug and error_samples:
        out["error_samples"] = error_samples
    return out


def _get_inbound_file(cur, report_type: str, business_date: Optional[date]) -> Tuple[int, date, str, str, bytes]:
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
        raise FastAPIHTTPException(status_code=404, detail={"error": f"no {report_type} PDF found for {bd}"})

    file_id, bd, filename, sha256, file_bytes = row
    if not sha256:
        sha256 = hashlib.sha256(file_bytes).hexdigest()
    return int(file_id), bd, str(filename), str(sha256), bytes(file_bytes)


# -----------------------------------------------------------------------------
# SalesClass parsing
# -----------------------------------------------------------------------------

_NOISE_PREFIXES = (
    "Sales by Class",
    "Average Cost",
    "For:",
    "Buck",
    "Extended",
    "Printed:",
)

_NOISE_CONTAINS = (
    "Reports Totals",
    "Group Totals",
    "Group B Totals",
)

_NOISE_SUFFIXES = ("DSSumClsG",)


def _clean_text(s: str) -> str:
    s = (s or "").replace("\r", "\n").replace("\t", " ")
    s = re.sub(r"<BLANK>", " ", s, flags=re.I)
    # normalize split negatives "- 1960.24%" -> "-1960.24%"
    s = re.sub(r"(?<=\s)-\s+(?=\d)", "-", s)
    # add spaces between glued tokens: "1.64HILLMAN" -> "1.64 HILLMAN"
    s = re.sub(r"(\d)([A-Za-z])", r"\1 \2", s)
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = " ".join(s.split())
    return s.strip()


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


def _try_parse_salesclass_row(buf: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single "Sales by Class Summary" row from a rolling buffer.
    Typical format:
      ext_price ext_cost profit gp_pct% ext_tax CLASS NAME ... price_plus_tax CLASSCODE pct_total

    Handles wrapped lines by using the rolling buffer.
    """
    s = _clean_text(buf)
    if not s:
        return None

    # skip obvious non-rows
    if s == "Class":
        return None
    if any(s.startswith(p) for p in _NOISE_PREFIXES):
        return None
    if any(x in s for x in _NOISE_CONTAINS):
        return None
    if any(s.endswith(p) for p in _NOISE_SUFFIXES):
        return None

    # must start with a number or "(" (negative via parens)
    if not re.match(r"^[\d(]", s):
        return None

    # pct_total is the last number on the line (e.g. 0.47)
    m = re.search(r"(?P<pct_total>\d[\d,]*\.\d{2})\s*$", s)
    if not m:
        return None
    pct_total = _dec(m.group("pct_total"))
    left = s[: m.start()].rstrip()

    # tail: price_plus_tax then optional class_code (may be glued)
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


def _row_required_ok(r: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate required fields BEFORE attempting DB insert.
    salesclass_daily has NOT NULL class_code; class_name should also be present.
    """
    cc = (r.get("class_code") or "").strip()
    cn = (r.get("class_name") or "").strip()
    if not cc:
        return False, "missing_class_code"
    if not cn:
        return False, "missing_class_name"
    return True, "ok"


def _extract_salesclass_rows(
    text: str,
    biz_date: date,
    source_file_id: int,
    source_sha256: str,
    *,
    debug: bool,
    max_skip_samples: int = 5,
    max_numeric_samples: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (rows, parse_stats). Malformed rows are skipped (not fatal).
    """
    t = (text or "").replace("\r", "\n").replace("\t", " ")
    lines = [ln.strip() for ln in t.split("\n")]

    rows: List[Dict[str, Any]] = []
    buf = ""
    started = False

    skipped_required = 0
    skipped_required_reasons: Dict[str, int] = {}
    skipped_required_samples: List[Dict[str, Any]] = []

    # Numeric regression instrumentation
    numeric_all_null = 0
    numeric_fail_samples: List[Dict[str, Any]] = []

    candidates_parsed = 0

    for ln in lines:
        if not ln:
            continue

        if not started:
            # Start after header once we see 'Class' or a numeric row
            if ln.strip() == "Class" or re.match(r"^[\d(]\d*\.\d{2}\s+\d", ln):
                started = True
            else:
                continue

        # Skip noise lines
        if any(ln.startswith(p) for p in _NOISE_PREFIXES):
            continue
        if any(x in ln for x in _NOISE_CONTAINS):
            continue
        if any(ln.endswith(p) for p in _NOISE_SUFFIXES):
            continue

        buf = (buf + " " + ln).strip()

        r = _try_parse_salesclass_row(buf)
        if r:
            candidates_parsed += 1

            # enrich
            r["business_date"] = biz_date
            r["source_file_id"] = source_file_id
            r["source_sha256"] = source_sha256

            # Numeric instrumentation: if all numeric fields are None, capture sample
            numeric_fields = ["ext_price", "pct_total", "ext_cost", "profit", "gp_pct", "ext_tax", "price_plus_tax"]
            if all(r.get(f) is None for f in numeric_fields):
                numeric_all_null += 1
                if debug and len(numeric_fail_samples) < max_numeric_samples:
                    numeric_fail_samples.append({"buf": _clean_text(buf)[:400]})

            ok, reason = _row_required_ok(r)
            if ok:
                rows.append(r)
            else:
                skipped_required += 1
                skipped_required_reasons[reason] = skipped_required_reasons.get(reason, 0) + 1
                if debug and len(skipped_required_samples) < max_skip_samples:
                    skipped_required_samples.append({"reason": reason, "buf": _clean_text(buf)[:400]})

            buf = ""
        else:
            # prevent runaway buffer from killing memory
            if len(buf) > 6000:
                buf = ln

    # last buffer
    if buf:
        r = _try_parse_salesclass_row(buf)
        if r:
            candidates_parsed += 1
            r["business_date"] = biz_date
            r["source_file_id"] = source_file_id
            r["source_sha256"] = source_sha256

            numeric_fields = ["ext_price", "pct_total", "ext_cost", "profit", "gp_pct", "ext_tax", "price_plus_tax"]
            if all(r.get(f) is None for f in numeric_fields):
                numeric_all_null += 1
                if debug and len(numeric_fail_samples) < max_numeric_samples:
                    numeric_fail_samples.append({"buf": _clean_text(buf)[:400]})

            ok, reason = _row_required_ok(r)
            if ok:
                rows.append(r)
            else:
                skipped_required += 1
                skipped_required_reasons[reason] = skipped_required_reasons.get(reason, 0) + 1
                if debug and len(skipped_required_samples) < max_skip_samples:
                    skipped_required_samples.append({"reason": reason, "buf": _clean_text(buf)[:400]})

    if not rows:
        sample = "\n".join(lines[:180])
        raise ValueError("Could not extract any valid salesclass rows. Sample:\n" + sample)

    stats: Dict[str, Any] = {
        "candidates_parsed": candidates_parsed,
        "rows_parsed": len(rows),
        "rows_skipped_required": skipped_required,
        "rows_skipped": skipped_required,  # compatibility alias
        "skipped_required_reasons": skipped_required_reasons,
        "numeric_all_null_rows": numeric_all_null,
        "ext_price_non_null": sum(1 for r in rows if r.get("ext_price") is not None),
    }

    if debug:
        if skipped_required_samples:
            stats["skipped_required_samples"] = skipped_required_samples
        if numeric_fail_samples:
            stats["numeric_fail_samples"] = numeric_fail_samples

    return rows, stats


def _delete_salesclass_day(cur, biz_date: date) -> Dict[str, int]:
    """
    Delete day from all relevant tables (idempotency).
    """
    out = {"daily": 0, "history": 0, "effective": 0}
    try:
        cur.execute("DELETE FROM analytics.salesclass_daily WHERE business_date = %s", (biz_date,))
        out["daily"] = getattr(cur, "rowcount", 0) or 0
    except Exception:
        pass
    try:
        cur.execute("DELETE FROM analytics.salesclass_history_daily WHERE business_date = %s", (biz_date,))
        out["history"] = getattr(cur, "rowcount", 0) or 0
    except Exception:
        pass
    try:
        cur.execute("DELETE FROM analytics.salesclass_effective_daily WHERE business_date = %s", (biz_date,))
        out["effective"] = getattr(cur, "rowcount", 0) or 0
    except Exception:
        pass
    return out


def _insert_salesclass_day(
    conn,
    cur,
    biz_date: date,
    rows: List[Dict[str, Any]],
    *,
    request_id: str,
    debug: bool,
) -> Dict[str, Any]:
    """
    Delete + insert for the day, skipping any rows that violate constraints.
    Entire operation is transactional; row-level insert failures are skipped.
    """
    try:
        conn.autocommit = False
    except Exception:
        pass

    deleted = _delete_salesclass_day(cur, biz_date)

    daily = _safe_insert_rows(cur, "analytics", "salesclass_daily", rows, request_id=request_id, debug=debug)
    history = _safe_insert_rows(cur, "analytics", "salesclass_history_daily", rows, request_id=request_id, debug=debug)
    effective = _safe_insert_rows(cur, "analytics", "salesclass_effective_daily", rows, request_id=request_id, debug=debug)

    # Commit transaction
    conn.commit()

    return {"deleted": deleted, "daily": daily, "history": history, "effective": effective}


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/parse/salesclass/latest")
def parse_salesclass_latest(
    request: Request,
    x_parse_token: Optional[str] = Header(default=None, alias="X-Parse-Token"),
    x_debug: Optional[str] = Header(default=None, alias="X-Debug"),
):
    request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
    debug = (x_debug or "").strip() in ("1", "true", "True", "yes", "YES")

    _require_token(x_parse_token)
    dsn = _database_url()

    try:
        with _db_connect(dsn) as conn:
            with conn.cursor() as cur:
                file_id, biz_date, filename, sha256, pdf_bytes = _get_inbound_file(cur, "salesclass", None)
                text = _pdf_to_text(pdf_bytes)
                rows, parse_stats = _extract_salesclass_rows(text, biz_date, file_id, sha256, debug=debug)

                db_stats = _insert_salesclass_day(conn, cur, biz_date, rows, request_id=request_id, debug=debug)

                inserted_daily = int(db_stats.get("daily", {}).get("inserted", 0) or 0)
                ok = inserted_daily > 0

                return {
                    "ok": ok,
                    "request_id": request_id,
                    "inserted_rows": inserted_daily,
                    "source": {"id": file_id, "business_date": biz_date.isoformat(), "filename": filename, "sha256": sha256},
                    "parse": parse_stats,
                    "db": db_stats,
                }
    except FastAPIHTTPException:
        raise
    except Exception as e:
        _log("salesclass.latest.failed", request_id=request_id, error=str(e))
        return _json_error(request_id, "exception", str(e), http_status=200)


@app.post("/parse/salesclass/by-date")
def parse_salesclass_by_date(
    request: Request,
    business_date: date = Query(..., description="YYYY-MM-DD"),
    x_parse_token: Optional[str] = Header(default=None, alias="X-Parse-Token"),
    x_debug: Optional[str] = Header(default=None, alias="X-Debug"),
):
    request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
    debug = (x_debug or "").strip() in ("1", "true", "True", "yes", "YES")

    _require_token(x_parse_token)
    dsn = _database_url()

    try:
        with _db_connect(dsn) as conn:
            with conn.cursor() as cur:
                file_id, biz_date, filename, sha256, pdf_bytes = _get_inbound_file(cur, "salesclass", business_date)

                try:
                    text = _pdf_to_text(pdf_bytes)
                except Exception as e:
                    _log("salesclass.pdf_extract_failed", request_id=request_id, business_date=biz_date.isoformat(), error=str(e))
                    return _json_error(request_id, "pdf_extract_failed", str(e), http_status=200)

                try:
                    rows, parse_stats = _extract_salesclass_rows(text, biz_date, file_id, sha256, debug=debug)
                except Exception as e:
                    _log("salesclass.parse_failed", request_id=request_id, business_date=biz_date.isoformat(), error=str(e))
                    return _json_error(request_id, "parse_failed", str(e), http_status=200)

                try:
                    db_stats = _insert_salesclass_day(conn, cur, biz_date, rows, request_id=request_id, debug=debug)
                except Exception as e:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    _log("salesclass.db_failed", request_id=request_id, business_date=biz_date.isoformat(), error=str(e))
                    return _json_error(request_id, "db_failed", str(e), http_status=200)

                inserted_daily = int(db_stats.get("daily", {}).get("inserted", 0) or 0)
                ok = inserted_daily > 0

                return {
                    "ok": ok,
                    "request_id": request_id,
                    "inserted_rows": inserted_daily,
                    "source": {"id": file_id, "business_date": biz_date.isoformat(), "filename": filename, "sha256": sha256},
                    "parse": parse_stats,
                    "db": db_stats,
                }
    except FastAPIHTTPException:
        raise
    except Exception as e:
        _log("salesclass.by_date.unhandled", request_id=request_id, business_date=business_date.isoformat(), error=str(e))
        return _json_error(request_id, "exception", str(e), http_status=200)


# NOTE: PayType / SalesStat endpoints are intentionally left in place so existing workflows don't 404.
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
