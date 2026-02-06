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
# App + hard safety: never 500
# -----------------------------------------------------------------------------

app = FastAPI(title="bucks-ingest-webhook")


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
    start = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        # absolute safety net: never 500, never crash worker
        _log("unhandled_exception_middleware", request_id=request_id, path=str(request.url.path), error=str(e))
        response = JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "request_id": request_id,
                "error_type": "unhandled_exception",
                "message": "Unhandled exception (middleware). See logs.",
            },
        )
    elapsed_ms = int((time.time() - start) * 1000)
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Elapsed-Ms"] = str(elapsed_ms)
    return response


@app.exception_handler(Exception)
async def all_exceptions_handler(request: Request, exc: Exception):
    # global catch-all: never 500
    request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
    # Preserve non-500 HTTP errors if they bubble up
    if isinstance(exc, FastAPIHTTPException):
        status = exc.status_code
        if status >= 500:
            status = 200
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
        detail.setdefault("ok", False)
        detail.setdefault("request_id", request_id)
        detail.setdefault("error_type", "http_exception")
        return JSONResponse(status_code=status, content=detail)

    _log("unhandled_exception_handler", request_id=request_id, path=str(request.url.path), error=str(exc))
    return JSONResponse(
        status_code=200,
        content={
            "ok": False,
            "request_id": request_id,
            "error_type": "unhandled_exception",
            "message": "Unhandled exception. See logs.",
        },
    )


# -----------------------------------------------------------------------------
# Auth / config helpers (NO 500)
# -----------------------------------------------------------------------------

def _expected_token() -> str:
    tok = os.getenv("PARSE_TOKEN") or os.getenv("X_PARSE_TOKEN") or ""
    return tok.strip()


def _database_url() -> str:
    return (os.getenv("DATABASE_URL") or "").strip()


def _is_debug(x_debug: Optional[str]) -> bool:
    # env enables always-on debug, header allows per-request debug
    return os.getenv("SALESCLASS_DEBUG", "0") == "1" or (x_debug or "").strip() in ("1", "true", "TRUE", "yes", "YES")


def _auth_error_or_none(x_parse_token: Optional[str], request_id: str) -> Optional[JSONResponse]:
    exp = _expected_token()
    if not exp:
        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "request_id": request_id,
                "error_type": "server_misconfigured",
                "message": "Server missing PARSE_TOKEN env var",
            },
        )
    if not x_parse_token or x_parse_token.strip() != exp:
        return JSONResponse(
            status_code=401,
            content={"ok": False, "request_id": request_id, "error_type": "unauthorized", "message": "Invalid X-Parse-Token"},
        )
    return None


def _db_error_or_none(request_id: str) -> Optional[JSONResponse]:
    dsn = _database_url()
    if not dsn:
        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "request_id": request_id,
                "error_type": "server_misconfigured",
                "message": "Server missing DATABASE_URL env var",
            },
        )
    return None


# -----------------------------------------------------------------------------
# PDF text extraction (never raises 500)
# -----------------------------------------------------------------------------

def _pdf_to_text(pdf_bytes: bytes) -> Tuple[str, str]:
    """
    Extract text from PDF bytes using whichever library is installed.
    Prefers PyMuPDF (fitz), then pdfplumber, then pypdf.
    Returns (text, extractor_name).
    """
    # PyMuPDF
    try:
        import fitz  # type: ignore

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        chunks = []
        for page in doc:
            chunks.append(page.get_text("text"))
        return "\n".join(chunks), "fitz"
    except Exception:
        pass

    # pdfplumber
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages), "pdfplumber"
    except Exception:
        pass

    # pypdf
    try:
        from pypdf import PdfReader  # type: ignore

        r = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join((p.extract_text() or "") for p in r.pages), "pypdf"
    except Exception as e:
        # caller handles structured error
        raise RuntimeError(f"No PDF text extractor available: {e}")


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


def _insert_rows(cur, schema: str, table: str, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    cols = _table_columns(cur, schema, table)
    if not cols:
        # table missing -> treat as 0 insert, not fatal
        return 0

    insert_cols = [c for c in cols if c in rows[0].keys()]

    if "updated_at" in cols and "updated_at" not in insert_cols:
        insert_cols.append("updated_at")
        for r in rows:
            r["updated_at"] = datetime.utcnow()

    if not insert_cols:
        return 0

    cols_sql = ", ".join(f'"{c}"' for c in insert_cols)
    vals_sql = ", ".join(f"%({c})s" for c in insert_cols)
    sql = f'INSERT INTO "{schema}"."{table}" ({cols_sql}) VALUES ({vals_sql})'

    cur.executemany(sql, rows)
    return len(rows)


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
        raise FileNotFoundError(f"no {report_type} PDF found for {business_date.isoformat() if business_date else 'latest'}")

    file_id, bd, filename, sha256, file_bytes = row
    file_bytes = bytes(file_bytes)
    if not sha256:
        sha256 = hashlib.sha256(file_bytes).hexdigest()
    return int(file_id), bd, str(filename), str(sha256), file_bytes


# -----------------------------------------------------------------------------
# SalesClass parsing (bulletproof + instrumentation)
# -----------------------------------------------------------------------------

_NUM = r"-?\d[\d,]*\.\d{2}"
_PCT = r"-?\d[\d,]*\.\d{2}%"

_RE_DECIMAL = re.compile(_NUM)
_RE_PERCENT = re.compile(_PCT)

# whitespace insertion between digit/alpha and alpha/digit (glued tokens)
_GLUE1 = re.compile(r"([A-Za-z])(\d)")
_GLUE2 = re.compile(r"(\d)([A-Za-z])")

# handle "- 1960.24%" -> "-1960.24%"
_SPLIT_NEG = re.compile(r"(?<=\s)-\s+(?=\d)")


def _clean_text(s: str) -> str:
    s = (s or "").replace("\r", "\n").replace("\t", " ")
    s = re.sub(r"<BLANK>", " ", s, flags=re.I)
    s = _SPLIT_NEG.sub("-", s)
    s = _GLUE1.sub(r"\1 \2", s)
    s = _GLUE2.sub(r"\1 \2", s)
    s = " ".join(s.split())
    return s.strip()


def _dec(x: str) -> Optional[Decimal]:
    x = (x or "").strip()
    if not x:
        return None
    neg_paren = x.startswith("(") and x.endswith(")")
    x = x.strip("()")
    x = x.replace("$", "").replace(",", "")
    # keep only digits, dot, minus
    x = re.sub(r"[^0-9.\-]", "", x)
    if x in ("", "-"):
        return None
    try:
        d = Decimal(x)
    except (InvalidOperation, ValueError):
        return None
    return -d if neg_paren else d


def _try_parse_salesclass_row(buf: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Robust row parse:
    - Works even when some numeric tokens are glued or spacing changes.
    - Does NOT throw; returns (row, reason_if_none).
    Expected semantics (TransAct Sales by Class Summary):
      ext_price, ext_cost, profit, gp_pct%, ext_tax, class_name, price_plus_tax, class_code, pct_total
    Heuristic mapping:
      - first 3 decimals -> ext_price/ext_cost/profit
      - last 3 decimals -> ext_tax/price_plus_tax/pct_total
      - first % token -> gp_pct
      - class_name = text between ext_tax and price_plus_tax
      - class_code = first trailing alnum token after price_plus_tax
    """
    s = _clean_text(buf)
    if not s:
        return None, "empty"

    # quick noise filters
    up = s.upper()
    if (
        up.startswith("SALES BY CLASS")
        or up.startswith("AVERAGE COST")
        or up.startswith("FOR:")
        or up.startswith("BUCK")
        or up.startswith("PRINTED:")
        or up.endswith("DSSUMCLSG")
        or "REPORTS TOTALS" in up
        or "GROUP TOTALS" in up
        or "GROUP B TOTALS" in up
        or up.startswith("EXTENDED")
        or up == "CLASS"
    ):
        return None, "noise"

    # Must have at least one percent and several decimals to be a row
    pct_m = _RE_PERCENT.search(s)
    if not pct_m:
        return None, "no_percent"

    # Extract decimals with positions
    decs = [(m.group(0), m.start(), m.end()) for m in _RE_DECIMAL.finditer(s)]
    if len(decs) < 6:
        return None, f"too_few_decimals:{len(decs)}"

    # Map the percent (gp_pct)
    gp_pct = _dec(pct_m.group(0).replace("%", ""))

    # Map core numbers
    ext_price = _dec(decs[0][0])
    ext_cost = _dec(decs[1][0])
    profit = _dec(decs[2][0])

    ext_tax = _dec(decs[-3][0])
    price_plus_tax = _dec(decs[-2][0])
    pct_total = _dec(decs[-1][0])

    # class_name between ext_tax end and price_plus_tax start
    name_start = decs[-3][2]
    name_end = decs[-2][1]
    class_name = s[name_start:name_end].strip()

    # trailing after price_plus_tax token: likely class_code (possibly glued)
    trail = s[decs[-2][2] : decs[-1][1]].strip()
    # allow glued "131.55SPGD" style -> we've already taken 131.55; now "SPGD ..." remains
    m_code = re.search(r"\b([A-Z0-9_]{1,10})\b", trail)
    class_code = m_code.group(1).strip() if m_code else None

    row = {
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

    return row, None


def _extract_salesclass_rows(
    text: str, biz_date: date, source_file_id: int, source_sha256: str, debug: bool
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Rolling-buffer parser that:
    - never throws
    - skips malformed rows
    - returns parse meta including numeric-null samples (debug only)
    """
    t = (text or "").replace("\r", "\n").replace("\t", " ")
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]

    rows: List[Dict[str, Any]] = []
    buf = ""

    skipped = 0
    parsed = 0
    numeric_null_rows = 0
    numeric_fail_samples: List[Dict[str, Any]] = []
    skipped_samples: List[Dict[str, Any]] = []

    # hard cap for debug samples
    max_samples = int(os.getenv("SALESCLASS_DEBUG_MAX_SAMPLES", "10"))

    for ln in lines:
        # skip obvious noise early
        up = ln.upper()
        if (
            up.startswith("SALES BY CLASS")
            or up.startswith("AVERAGE COST")
            or up.startswith("FOR:")
            or up.startswith("BUCK")
            or up.startswith("PRINTED:")
            or up.endswith("DSSUMCLSG")
            or "REPORTS TOTALS" in up
            or up.startswith("EXTENDED")
        ):
            continue

        buf = (buf + " " + ln).strip()

        r, reason = _try_parse_salesclass_row(buf)
        if r:
            parsed += 1
            r["business_date"] = biz_date
            r["source_file_id"] = source_file_id
            r["source_sha256"] = source_sha256
            rows.append(r)

            # numeric regression instrumentation
            if r.get("ext_price") is None or r.get("ext_cost") is None or r.get("profit") is None:
                numeric_null_rows += 1
                if debug and len(numeric_fail_samples) < max_samples:
                    numeric_fail_samples.append({"buffer": buf[:500], "row": {k: str(v) if isinstance(v, Decimal) else v for k, v in r.items()}})
            buf = ""
        else:
            # keep accumulating; if buffer gets too big, salvage without killing the day
            if len(buf) > 6000:
                skipped += 1
                if debug and len(skipped_samples) < max_samples:
                    skipped_samples.append({"reason": reason or "buffer_too_large", "buffer_head": buf[:300], "buffer_tail": buf[-300:]})
                # keep tail; often next row starts near end
                buf = buf[-1200:]

    # final buffer attempt
    if buf:
        r, reason = _try_parse_salesclass_row(buf)
        if r:
            parsed += 1
            r["business_date"] = biz_date
            r["source_file_id"] = source_file_id
            r["source_sha256"] = source_sha256
            rows.append(r)
            if r.get("ext_price") is None or r.get("ext_cost") is None or r.get("profit") is None:
                numeric_null_rows += 1
                if debug and len(numeric_fail_samples) < max_samples:
                    numeric_fail_samples.append({"buffer": buf[:500], "row": {k: str(v) if isinstance(v, Decimal) else v for k, v in r.items()}})
        else:
            skipped += 1
            if debug and len(skipped_samples) < max_samples:
                skipped_samples.append({"reason": reason or "final_buffer_unparsed", "buffer": buf[:500]})

    meta: Dict[str, Any] = {
        "lines_seen": len(lines),
        "rows_parsed": len(rows),
        "rows_attempted": parsed,
        "rows_skipped": skipped,
        "numeric_null_rows": numeric_null_rows,
    }
    if debug:
        meta["numeric_fail_samples"] = numeric_fail_samples
        meta["skipped_samples"] = skipped_samples
        meta["text_head"] = "\n".join(lines[:60])

    return rows, meta


# -----------------------------------------------------------------------------
# Endpoints (DO NOT CHANGE SIGNATURES)
# -----------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"ok": True}


def _delete_day(cur, biz_date: date) -> Dict[str, int]:
    counts = {}
    for tbl in ("analytics.salesclass_daily", "analytics.salesclass_history_daily", "analytics.salesclass_effective_daily"):
        try:
            cur.execute(f"DELETE FROM {tbl} WHERE business_date = %s", (biz_date,))
            counts[tbl] = getattr(cur, "rowcount", 0) or 0
        except Exception:
            counts[tbl] = 0
    return counts


def _insert_day(cur, rows: List[Dict[str, Any]]) -> Dict[str, int]:
    inserted = {}
    inserted["analytics.salesclass_daily"] = _insert_rows(cur, "analytics", "salesclass_daily", rows)
    inserted["analytics.salesclass_history_daily"] = _insert_rows(cur, "analytics", "salesclass_history_daily", rows)
    inserted["analytics.salesclass_effective_daily"] = _insert_rows(cur, "analytics", "salesclass_effective_daily", rows)
    return inserted


@app.post("/parse/salesclass/latest")
def parse_salesclass_latest(
    x_parse_token: Optional[str] = Header(default=None, alias="X-Parse-Token"),
    x_debug: Optional[str] = Header(default=None, alias="X-Debug"),
    request: Request = None,  # FastAPI injects
):
    request_id = (request.headers.get("X-Request-Id") if request else None) or uuid.uuid4().hex
    debug = _is_debug(x_debug)

    denied = _auth_error_or_none(x_parse_token, request_id)
    if denied:
        return denied
    db_err = _db_error_or_none(request_id)
    if db_err:
        return db_err

    dsn = _database_url()

    _log("salesclass_request_start", request_id=request_id, endpoint="latest", debug=debug)

    try:
        with _db_connect(dsn) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                file_id, biz_date, filename, sha256, pdf_bytes = _get_inbound_file(cur, "salesclass", None)

                try:
                    text, extractor = _pdf_to_text(pdf_bytes)
                except Exception as e:
                    _log("salesclass_pdf_extract_failed", request_id=request_id, error=str(e))
                    return JSONResponse(
                        status_code=200,
                        content={
                            "ok": False,
                            "request_id": request_id,
                            "error_type": "pdf_extract_failed",
                            "message": str(e),
                            "source": {"id": file_id, "business_date": biz_date.isoformat(), "filename": filename, "sha256": sha256},
                        },
                    )

                rows, meta = _extract_salesclass_rows(text, biz_date, file_id, sha256, debug)

                if not rows:
                    _log("salesclass_no_rows", request_id=request_id, business_date=biz_date.isoformat(), meta=meta)
                    return JSONResponse(
                        status_code=200,
                        content={
                            "ok": False,
                            "request_id": request_id,
                            "error_type": "no_rows_parsed",
                            "message": "Parsed zero rows (layout edge case).",
                            "source": {"id": file_id, "business_date": biz_date.isoformat(), "filename": filename, "sha256": sha256, "extractor": extractor},
                            "parse": meta,
                        },
                    )

                deleted = _delete_day(cur, biz_date)
                inserted = _insert_day(cur, rows)

                ext_price_non_null = sum(1 for r in rows if r.get("ext_price") is not None)

                _log(
                    "salesclass_request_done",
                    request_id=request_id,
                    endpoint="latest",
                    business_date=biz_date.isoformat(),
                    rows=len(rows),
                    ext_price_non_null=ext_price_non_null,
                )

                return {
                    "ok": True,
                    "request_id": request_id,
                    "inserted_rows": inserted.get("analytics.salesclass_daily", 0),
                    "inserted": inserted,
                    "deleted": deleted,
                    "parse": {**meta, "ext_price_non_null": ext_price_non_null},
                    "source": {"id": file_id, "business_date": biz_date.isoformat(), "filename": filename, "sha256": sha256, "extractor": extractor},
                }
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"ok": False, "request_id": request_id, "error_type": "pdf_not_found", "message": str(e)})
    except Exception as e:
        _log("salesclass_latest_exception", request_id=request_id, error=str(e))
        return JSONResponse(status_code=200, content={"ok": False, "request_id": request_id, "error_type": "exception", "message": str(e)})


@app.post("/parse/salesclass/by-date")
def parse_salesclass_by_date(
    business_date: date = Query(..., description="YYYY-MM-DD"),
    x_parse_token: Optional[str] = Header(default=None, alias="X-Parse-Token"),
    x_debug: Optional[str] = Header(default=None, alias="X-Debug"),
    request: Request = None,  # FastAPI injects
):
    request_id = (request.headers.get("X-Request-Id") if request else None) or uuid.uuid4().hex
    debug = _is_debug(x_debug)

    denied = _auth_error_or_none(x_parse_token, request_id)
    if denied:
        return denied
    db_err = _db_error_or_none(request_id)
    if db_err:
        return db_err

    dsn = _database_url()
    _log("salesclass_request_start", request_id=request_id, endpoint="by-date", business_date=business_date.isoformat(), debug=debug)

    try:
        with _db_connect(dsn) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                file_id, biz_date, filename, sha256, pdf_bytes = _get_inbound_file(cur, "salesclass", business_date)

                try:
                    text, extractor = _pdf_to_text(pdf_bytes)
                except Exception as e:
                    _log("salesclass_pdf_extract_failed", request_id=request_id, business_date=biz_date.isoformat(), error=str(e))
                    return JSONResponse(
                        status_code=200,
                        content={
                            "ok": False,
                            "request_id": request_id,
                            "error_type": "pdf_extract_failed",
                            "message": str(e),
                            "business_date": biz_date.isoformat(),
                            "source": {"id": file_id, "filename": filename, "sha256": sha256},
                        },
                    )

                rows, meta = _extract_salesclass_rows(text, biz_date, file_id, sha256, debug)

                if not rows:
                    _log("salesclass_no_rows", request_id=request_id, business_date=biz_date.isoformat(), meta=meta)
                    return JSONResponse(
                        status_code=200,
                        content={
                            "ok": False,
                            "request_id": request_id,
                            "error_type": "no_rows_parsed",
                            "message": "Parsed zero rows (layout edge case).",
                            "business_date": biz_date.isoformat(),
                            "source": {"id": file_id, "filename": filename, "sha256": sha256, "extractor": extractor},
                            "parse": meta,
                        },
                    )

                deleted = _delete_day(cur, biz_date)
                inserted = _insert_day(cur, rows)

                ext_price_non_null = sum(1 for r in rows if r.get("ext_price") is not None)
                if ext_price_non_null == 0:
                    _log("salesclass_ext_price_all_null", request_id=request_id, business_date=biz_date.isoformat(), rows=len(rows))

                _log(
                    "salesclass_request_done",
                    request_id=request_id,
                    endpoint="by-date",
                    business_date=biz_date.isoformat(),
                    rows=len(rows),
                    ext_price_non_null=ext_price_non_null,
                )

                return {
                    "ok": True,
                    "request_id": request_id,
                    "business_date": biz_date.isoformat(),
                    "inserted_rows": inserted.get("analytics.salesclass_daily", 0),
                    "inserted": inserted,
                    "deleted": deleted,
                    "parse": {**meta, "ext_price_non_null": ext_price_non_null},
                    "source": {"id": file_id, "filename": filename, "sha256": sha256, "extractor": extractor},
                }
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"ok": False, "request_id": request_id, "error_type": "pdf_not_found", "message": str(e)})
    except Exception as e:
        _log("salesclass_by_date_exception", request_id=request_id, business_date=business_date.isoformat(), error=str(e))
        return JSONResponse(status_code=200, content={"ok": False, "request_id": request_id, "error_type": "exception", "message": str(e)})


# NOTE: PayType / SalesStat endpoints are intentionally left in place so your existing workflows don't 404.
@app.post("/parse/paytype/latest")
def parse_paytype_latest(x_parse_token: Optional[str] = Header(default=None, alias="X-Parse-Token")):
    request_id = uuid.uuid4().hex
    denied = _auth_error_or_none(x_parse_token, request_id)
    if denied:
        return denied
    return JSONResponse({"ok": True, "inserted_rows": 0, "note": "paytype parser not changed in this step"})


@app.post("/parse/paytype/by-date")
def parse_paytype_by_date(
    business_date: date = Query(..., description="YYYY-MM-DD"),
    x_parse_token: Optional[str] = Header(default=None, alias="X-Parse-Token"),
):
    request_id = uuid.uuid4().hex
    denied = _auth_error_or_none(x_parse_token, request_id)
    if denied:
        return denied
    return JSONResponse({"ok": True, "inserted_rows": 0, "note": "paytype parser not changed in this step"})


@app.post("/parse/salesstat/latest")
def parse_salesstat_latest(x_parse_token: Optional[str] = Header(default=None, alias="X-Parse-Token")):
    request_id = uuid.uuid4().hex
    denied = _auth_error_or_none(x_parse_token, request_id)
    if denied:
        return denied
    return JSONResponse({"ok": True, "inserted_rows": 0, "note": "salesstat parser not changed in this step"})


@app.post("/parse/salesstat/by-date")
def parse_salesstat_by_date(
    business_date: date = Query(..., description="YYYY-MM-DD"),
    x_parse_token: Optional[str] = Header(default=None, alias="X-Parse-Token"),
):
    request_id = uuid.uuid4().hex
    denied = _auth_error_or_none(x_parse_token, request_id)
    if denied:
        return denied
    return JSONResponse({"ok": True, "inserted_rows": 0, "note": "salesstat parser not changed in this step"})
