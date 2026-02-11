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
# App + hard safety: never crash worker; structured JSON errors
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
        # last-resort catch: never crash the worker
        _log(
            "request.unhandled_exception",
            request_id=request_id,
            path=str(request.url.path),
            query=str(request.url.query),
            method=request.method,
            error=str(e),
        )
        response = _json_error(request_id, "exception", str(e), http_status=200)

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

    # never emit 5xx from our app
    http_status = status if status < 500 else 200

    detail = getattr(exc, "detail", None)
    extra: Optional[Dict[str, Any]] = None
    if isinstance(detail, dict):
        message = detail.get("error") or detail.get("message") or str(detail)
        extra = {k: v for k, v in detail.items() if k not in ("error", "message")}
    else:
        message = str(detail) if detail else str(exc)

    et = "http_error"
    if status == 401:
        et = "unauthorized"
    elif status == 404:
        et = "not_found"
    elif status == 409:
        et = "conflict"

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
    try:
        import fitz  # type: ignore

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        chunks = []
        for page in doc:
            chunks.append(page.get_text("text"))
        return "\n".join(chunks)
    except Exception:
        pass

    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception:
        pass

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
    if not rows:
        return None, None, None

    cols = _table_columns(cur, schema, table)
    if not cols:
        return None, None, None

    insert_cols = [c for c in cols if c in rows[0].keys()]

    if "updated_at" in cols and "updated_at" not in insert_cols:
        insert_cols.append("updated_at")
        for r in rows:
            r["updated_at"] = datetime.utcnow()

    if not insert_cols:
        return None, None, None

    cols_sql = ", ".join(f'"{c}"' for c in insert_cols)
    vals_sql = ", ".join(f"%({c})s" for c in insert_cols)
    sql = f'INSERT INTO "{schema}"."{table}" ({cols_sql}) VALUES ({vals_sql})'
    return sql, insert_cols, cols


def _safe_exec(
    cur,
    sql: str,
    params: Tuple[Any, ...],
    *,
    sp_name: str,
    request_id: str,
    debug: bool,
) -> Tuple[bool, int, Optional[str]]:
    """
    psycopg2 will mark the transaction as aborted on *any* statement error.
    Catching doesn't fix it — you must ROLLBACK. We use SAVEPOINT for each statement.
    """
    try:
        cur.execute(f"SAVEPOINT {sp_name}")
    except Exception:
        return False, 0, "savepoint_failed"

    try:
        cur.execute(sql, params)
        rc = getattr(cur, "rowcount", 0) or 0
        cur.execute(f"RELEASE SAVEPOINT {sp_name}")
        return True, int(rc), None
    except Exception as e:
        err = str(e)
        try:
            cur.execute(f"ROLLBACK TO SAVEPOINT {sp_name}")
            cur.execute(f"RELEASE SAVEPOINT {sp_name}")
        except Exception:
            pass

        _log(
            "db.statement_failed",
            request_id=request_id,
            sp=sp_name,
            error=err,
            sql=sql,
        )
        return False, 0, err


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
    Insert rows without aborting the day:
    - SAVEPOINT per row
    - skip DB/constraint failures
    """
    out: Dict[str, Any] = {"schema": schema, "table": table, "inserted": 0, "skipped_db": 0}
    if not rows:
        return out

    sql, insert_cols, _cols = _prepare_insert(cur, schema, table, rows)
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
# SalesClass parsing (token-stream primary; glue-tolerant)
# -----------------------------------------------------------------------------

_NOISE_PREFIXES = (
    "Sales by Class",
    "Average Cost",
    "For:",
    "Buck",
    "Extended",
    "Printed:",
)
_NOISE_CONTAINS = ("Reports Totals", "Group Totals", "Group B Totals")
_NOISE_SUFFIXES = ("DSSumClsG", "DSSumCls")

_DEC2 = r"-?\d[\d,]*\.\d{2}"


def _clean_text(s: str) -> str:
    s = (s or "").replace("\r", "\n").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # normalize "- 1960.24%" -> "-1960.24%"
    s = re.sub(r"(?<=\s)-\s+(?=\d)", "-", s)

    # insert boundaries: "1.64HILLMAN" -> "1.64 HILLMAN"
    s = re.sub(r"(\d)([A-Za-z])", r"\1 \2", s)
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)

    # split glued decimals: "59.4564.431.27" -> "59.45 64.43 1.27"
    s = re.sub(r"(\d\.\d{2})(?=\d)", r"\1 ", s)

    # split decimal followed by minus sign: "5.00-5.00" -> "5.00 -5.00"
    s = re.sub(r"(\d\.\d{2})(?=-)", r"\1 ", s)

    return " ".join(s.split()).strip()


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


def _is_class_code(tok: str) -> bool:
    if not tok:
        return False
    if tok.upper() == "<BLANK>":
        return True
    if re.fullmatch(r"\d{1,4}", tok):
        return True
    if re.fullmatch(r"[A-Z_][A-Z0-9_]{0,10}", tok):
        return True
    return False


def _parse_salesclass_token_stream(tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Works with the real PDFs (01/27 sample):
      class_code, ext_price, ext_cost, profit, gp_pct%, ext_tax, [class_name...], sales_per, price_plus_tax, pct_total

    Notes:
    - class_name can be empty
    - last 3 fields are always 3 consecutive decimals
    """
    dec2 = re.compile(rf"^{_DEC2}$")

    # Find first plausible row start
    start: Optional[int] = None
    for i in range(len(tokens) - 6):
        if _is_class_code(tokens[i]) and dec2.match(tokens[i + 1] or ""):
            start = i
            break
    if start is None:
        return []

    rows: List[Dict[str, Any]] = []
    i = start

    while i < len(tokens) - 9:
        cc = tokens[i]
        if not _is_class_code(cc):
            i += 1
            continue

        # ext_price ext_cost profit gp% ext_tax must be present
        if not dec2.match(tokens[i + 1]):
            i += 1
            continue
        if not dec2.match(tokens[i + 2]):
            i += 1
            continue
        if not dec2.match(tokens[i + 3]):
            i += 1
            continue

        gp_tok = tokens[i + 4]
        if not gp_tok.endswith("%"):
            i += 1
            continue

        ext_tax_tok = tokens[i + 5]
        if not dec2.match(ext_tax_tok):
            i += 1
            continue

        # find 3 consecutive decimals (sales_per, price_plus_tax, pct_total)
        j = i + 6
        found = False
        while j < len(tokens) - 2:
            if dec2.match(tokens[j]) and dec2.match(tokens[j + 1]) and dec2.match(tokens[j + 2]):
                sales_per_tok = tokens[j]
                price_plus_tax_tok = tokens[j + 1]
                pct_total_tok = tokens[j + 2]
                class_name = " ".join(tokens[i + 6 : j]).strip()

                rows.append(
                    {
                        "class_code": "<BLANK>" if cc.upper() == "<BLANK>" else cc,
                        "class_name": class_name,
                        "ext_price": _dec(tokens[i + 1]),
                        "ext_cost": _dec(tokens[i + 2]),
                        "profit": _dec(tokens[i + 3]),
                        "gp_pct": _dec(gp_tok.replace("%", "")),
                        "ext_tax": _dec(ext_tax_tok),
                        "sales_per": _dec(sales_per_tok),
                        "price_plus_tax": _dec(price_plus_tax_tok),
                        "pct_total": _dec(pct_total_tok),
                        "_format": "token_stream",
                    }
                )

                i = j + 3
                found = True
                break

            # safety: if we hit another class_code before tail numbers, treat as broken row and resync
            if _is_class_code(tokens[j]) and j > i + 6:
                i = j
                found = True
                break

            j += 1

        if not found:
            break

    return rows


def _try_parse_salesclass_row(buf: str) -> Optional[Dict[str, Any]]:
    """
    Legacy line-buffer parser kept only as fallback.
    """
    s = _clean_text(buf)
    if not s:
        return None
    if s == "Class":
        return None
    if any(s.startswith(p) for p in _NOISE_PREFIXES):
        return None
    if any(x in s for x in _NOISE_CONTAINS):
        return None
    if any(s.endswith(p) for p in _NOISE_SUFFIXES):
        return None

    # If a full row is present on one line, token-stream logic can still parse it.
    toks = s.split()
    parsed = _parse_salesclass_token_stream(toks)
    if parsed:
        # return only the first row parsed from this buffer
        return parsed[0]
    return None


def _row_required_ok(r: Dict[str, Any]) -> Tuple[bool, str]:
    cc = (r.get("class_code") or "").strip()
    if not cc:
        return False, "missing_class_code"
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
    # Primary: token-stream parse (works for 01/27 PDF you uploaded)
    cleaned = _clean_text(text or "")
    tokens = cleaned.split()
    stream_rows = _parse_salesclass_token_stream(tokens)

    rows: List[Dict[str, Any]] = []
    candidates_parsed = 0
    skipped_required = 0
    skipped_required_reasons: Dict[str, int] = {}
    skipped_required_samples: List[Dict[str, Any]] = []
    numeric_all_null = 0
    numeric_fail_samples: List[Dict[str, Any]] = []

    fmt_token = 0
    fmt_fallback = 0

    if stream_rows:
        for r in stream_rows:
            candidates_parsed += 1

            numeric_fields = [
                "ext_price",
                "pct_total",
                "ext_cost",
                "profit",
                "gp_pct",
                "ext_tax",
                "price_plus_tax",
                "sales_per",
            ]
            if all(r.get(f) is None for f in numeric_fields if f not in ("sales_per",)):
                numeric_all_null += 1
                if debug and len(numeric_fail_samples) < max_numeric_samples:
                    numeric_fail_samples.append({"row": {k: r.get(k) for k in ("class_code", "class_name")}})

            ok, reason = _row_required_ok(r)
            if ok:
                r["business_date"] = biz_date
                r["source_file_id"] = source_file_id
                r["source_sha256"] = source_sha256
                fmt_token += 1
                r.pop("_format", None)
                rows.append(r)
            else:
                skipped_required += 1
                skipped_required_reasons[reason] = skipped_required_reasons.get(reason, 0) + 1
                if debug and len(skipped_required_samples) < max_skip_samples:
                    skipped_required_samples.append({"reason": reason, "row": r})
    else:
        # Fallback: line buffer mode (kept for safety)
        t = (text or "").replace("\r", "\n").replace("\t", " ")
        lines = [ln.strip() for ln in t.split("\n")]
        buf = ""
        started = False

        for ln in lines:
            if not ln:
                continue

            if not started:
                if ln.strip() == "Class":
                    started = True
                else:
                    lnc = _clean_text(ln)
                    if lnc.startswith("<BLANK>") or re.match(r"^([A-Z_][A-Z0-9_]{0,10}|\d{1,4})\s+", lnc):
                        started = True
                    else:
                        continue

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
                ok, reason = _row_required_ok(r)
                if ok:
                    r["business_date"] = biz_date
                    r["source_file_id"] = source_file_id
                    r["source_sha256"] = source_sha256
                    fmt_fallback += 1
                    r.pop("_format", None)
                    rows.append(r)
                else:
                    skipped_required += 1
                    skipped_required_reasons[reason] = skipped_required_reasons.get(reason, 0) + 1
                    if debug and len(skipped_required_samples) < max_skip_samples:
                        skipped_required_samples.append({"reason": reason, "buf": _clean_text(buf)[:400]})
                buf = ""
            else:
                if len(buf) > 6000:
                    buf = ln

    if not rows:
        sample = "\n".join((text or "").splitlines()[:180])
        raise ValueError("Could not extract any valid salesclass rows. Sample:\n" + sample)

    stats: Dict[str, Any] = {
        "candidates_parsed": candidates_parsed,
        "rows_parsed": len(rows),
        "rows_skipped_required": skipped_required,
        "rows_skipped": skipped_required,
        "skipped_required_reasons": skipped_required_reasons,
        "numeric_all_null_rows": numeric_all_null,
        "ext_price_non_null": sum(1 for r in rows if r.get("ext_price") is not None),
        "format_token_stream_rows": fmt_token,
        "format_fallback_rows": fmt_fallback,
    }

    if debug:
        if skipped_required_samples:
            stats["skipped_required_samples"] = skipped_required_samples
        if numeric_fail_samples:
            stats["numeric_fail_samples"] = numeric_fail_samples

    return rows, stats


def _delete_salesclass_day(cur, biz_date: date, *, request_id: str, debug: bool) -> Dict[str, Any]:
    """
    Delete only from analytics.salesclass_daily.

    IMPORTANT:
    - salesclass_effective_daily is a view (not updatable)
    - salesclass_history_daily has different constraints/columns (net_sales NOT NULL)
    We do NOT touch those in this endpoint.
    """
    out: Dict[str, Any] = {"daily": 0}
    errors: List[Dict[str, Any]] = []

    sql = 'DELETE FROM "analytics"."salesclass_daily" WHERE business_date = %s'
    ok, rc, err = _safe_exec(cur, sql, (biz_date,), sp_name="sp_del_daily", request_id=request_id, debug=debug)
    if ok:
        out["daily"] = rc
    else:
        if debug and err:
            errors.append({"table": "analytics.salesclass_daily", "error": err})

    if debug and errors:
        out["delete_errors"] = errors
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
    # ensure clean transaction state
    try:
        conn.rollback()
    except Exception:
        pass

    try:
        conn.autocommit = False
    except Exception:
        pass

    deleted = _delete_salesclass_day(cur, biz_date, request_id=request_id, debug=debug)

    # Insert ONLY the daily table
    daily = _safe_insert_rows(cur, "analytics", "salesclass_daily", rows, request_id=request_id, debug=debug)

    conn.commit()
    return {
        "deleted": deleted,
        "daily": daily,
        "note": "history/effective not written (history needs net_sales; effective is a DISTINCT view)",
    }


# -----------------------------------------------------------------------------
# Endpoints (do not change signatures)
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


# PayType / SalesStat placeholders (unchanged)
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
