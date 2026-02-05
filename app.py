# ===================== SALESCLASS PARSER (ROBUST) =====================
import re
from decimal import Decimal, InvalidOperation

_SA_MONEY2_RE = re.compile(r'^\(?-?[\d,]+\.\d{2}')
_SA_NUM_RE    = re.compile(r'^\(?-?[\d,]+(?:\.\d+)?\)?')

def _sa_to_decimal(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    # strip everything except digits, dot, minus
    s = re.sub(r"[^0-9\.\-]", "", s)
    if s in ("", "-", "."):
        return None

    try:
        v = Decimal(s)
    except InvalidOperation:
        return None

    return (-v if neg else v)

def _sa_money2_and_suffix(tok):
    """
    Split token into (money_value, suffix).
    Handles tokens like:
      21.22116    -> money=21.22, suffix="116"
      28.78AUTO   -> money=28.78, suffix="AUTO"
      10.49<BLANK>-> money=10.49, suffix="<BLANK>"
    """
    if tok is None:
        return None, ""
    t = str(tok).strip()
    if t == "":
        return None, ""

    t_nocomma = t.replace(",", "")
    m = _SA_MONEY2_RE.match(t_nocomma)
    if not m:
        return None, t

    money_txt = m.group(0)
    suffix = t_nocomma[len(money_txt):]
    return _sa_to_decimal(money_txt), suffix

def _sa_is_numlike(tok):
    if tok is None:
        return False
    t = str(tok).strip()
    return bool(_SA_NUM_RE.match(t.replace(",", "")))

def _sa_merge_broken_tokens(tokens):
    """
    Fix common PDF-extraction splits like:
      gp% token: "-" then "1960.24%"  -> "-1960.24%"
    """
    out = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "-" and i + 1 < len(tokens) and str(tokens[i + 1]).endswith("%"):
            out.append("-" + str(tokens[i + 1]))
            i += 2
            continue
        if str(t).endswith("-") and i + 1 < len(tokens) and str(tokens[i + 1]).endswith("%"):
            out.append(str(t) + str(tokens[i + 1]))
            i += 2
            continue
        out.append(t)
        i += 1
    return out

def _sa_parse_salesclass_row_from_buffer(buf):
    """
    Returns row dict for analytics.salesclass_daily, or None if buffer doesn't look like a row.
    Expected logical layout:
      ext_price ext_cost profit gp_pct ext_tax class_name price_plus_tax+class_code pct_total

    BUT tokens may be glued:
      ext_tax like "1.64HILLMAN"
      price_plus_tax+code like "21.22116" or "28.78AUTO"
    """
    if not buf:
        return None

    s = " ".join(str(buf).replace("\r", "\n").split())
    if not s:
        return None

    # Skip obvious totals/footers
    up = s.upper()
    if "PRINTED:" in up or "REPORTS TOTALS" in up or "GROUP TOTALS" in up or "TOTALS" in up:
        return None

    tokens = _sa_merge_broken_tokens(s.split())
    if len(tokens) < 7:
        return None

    # First 4 should be numbers (gp has % usually)
    if not (_sa_is_numlike(tokens[0]) and _sa_is_numlike(tokens[1]) and _sa_is_numlike(tokens[2])):
        return None
    if not (_sa_is_numlike(str(tokens[3]).replace("%", ""))):
        return None
    if not _sa_is_numlike(tokens[-1]):
        return None

    ext_price = _sa_to_decimal(tokens[0])
    ext_cost  = _sa_to_decimal(tokens[1])
    profit    = _sa_to_decimal(tokens[2])
    gp_pct    = _sa_to_decimal(str(tokens[3]).replace("%", ""))

    # ext_tax token may contain glued class_name prefix (e.g., 1.64HILLMAN)
    ext_tax_val, tax_suffix = _sa_money2_and_suffix(tokens[4])
    tax_suffix = (tax_suffix or "").strip()

    # Determine where price+tax / class code lives
    pct_total = _sa_to_decimal(tokens[-1])

    # Most common: second-to-last token contains price_plus_tax + class_code
    price_plus_tax_val, code_suffix = _sa_money2_and_suffix(tokens[-2])
    code_suffix = (code_suffix or "").strip()

    # If that didn't produce money, assume tokens[-3] is price and tokens[-2] is class_code
    idx_name_end = len(tokens) - 2
    class_code = code_suffix

    if price_plus_tax_val is None:
        if len(tokens) < 8:
            return None
        price_plus_tax_val = _sa_to_decimal(tokens[-3])
        class_code = str(tokens[-2]).strip()
        idx_name_end = len(tokens) - 3

    # Class name is everything between ext_tax token and price token, plus any glued tax suffix
    name_tokens = []
    if tax_suffix:
        name_tokens.append(tax_suffix)

    middle = tokens[5:idx_name_end]
    if middle:
        name_tokens.extend([str(x) for x in middle])

    class_name = " ".join([t for t in name_tokens if t]).strip()

    # Normalize blanks
    if class_code in ("", "<BLANK>", "BLANK"):
        class_code = "UNCAT"
    if class_name == "" and class_code == "UNCAT":
        class_name = "Uncategorized"

    # If ext_price is missing but price_plus_tax/ext_tax exist, derive it
    if ext_price is None and price_plus_tax_val is not None and ext_tax_val is not None:
        ext_price = price_plus_tax_val - ext_tax_val

    # Last sanity: ext_price must exist for your dashboards; if still None, drop row
    if ext_price is None:
        return None

    return {
        "business_date": None,  # caller sets
        "class_code": class_code,
        "class_name": class_name,
        "ext_price": ext_price,
        "pct_total": pct_total,
        "ext_cost": ext_cost,
        "profit": profit,
        "gp_pct": gp_pct,
        "ext_tax": ext_tax_val,
        "price_plus_tax": price_plus_tax_val,
        "source_file_id": None,     # caller sets
        "source_sha256": None,      # caller sets
    }

def _extract_salesclass_rows(text, biz_date, source_file_id, source_sha256):
    """
    Drop-in replacement for the existing salesclass row extractor.

    - Handles glued tokens (e.g., 1.64HILLMAN, 21.22116)
    - Prevents ext_price from being NULL for valid rows
    - Assigns UNCAT when class code is blank, instead of crashing inserts
    """
    rows = []
    buf = ""

    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Stop at hard footer markers
        up = line.upper()
        if up.startswith("PRINTED:") or "REPORTS TOTALS" in up:
            # flush any pending buffer then stop
            if buf:
                r = _sa_parse_salesclass_row_from_buffer(buf)
                if r:
                    rows.append(r)
            buf = ""
            break

        # Build up a buffer; PDF extraction sometimes breaks one logical row across lines
        if buf:
            buf = buf + " " + line
        else:
            buf = line

        r = _sa_parse_salesclass_row_from_buffer(buf)
        if r:
            r["business_date"] = biz_date
            r["source_file_id"] = source_file_id
            r["source_sha256"] = source_sha256
            rows.append(r)
            buf = ""

        # Guardrail: if buffer gets too long, reset (prevents snowballing)
        if len(buf) > 1500:
            buf = ""

    # final flush
    if buf:
        r = _sa_parse_salesclass_row_from_buffer(buf)
        if r:
            r["business_date"] = biz_date
            r["source_file_id"] = source_file_id
            r["source_sha256"] = source_sha256
            rows.append(r)

    if not rows:
        sample = "\n".join(str(text).splitlines()[:120])
        raise ValueError("Could not extract any salesclass rows. Sample:\n" + sample)

    return rows
# =================== END SALESCLASS PARSER (ROBUST) ===================
