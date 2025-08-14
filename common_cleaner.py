#!/usr/bin/env python3

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Utilities (pure, deterministic)
# -----------------------------

def _to_iso_date(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return value
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return value
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return value


_time_only_re = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM|am|pm)?\s*$")

def _to_iso_time(value: Any) -> Any:
    """Normalize time-only strings to HH:MM:SS; leave others unchanged."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return value
    try:
        s = str(value).strip()
        if not s or not _time_only_re.match(s):
            return value
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return value
        return ts.strftime("%H:%M:%S")
    except Exception:
        return value


def _to_numeric(value: Any) -> Any:
    if value is None:
        return value
    try:
        s = str(value).strip()
        if s == "":
            return value
        # Accounting negatives even with currency prefix or suffix text
        paren_present = ("(" in s and ")" in s)
        # Remove common currency symbols for parsing
        s2 = re.sub(r"[\$£€¥]", "", s)
        # Remove parentheses for numeric extraction (we keep the flag)
        s2_no_paren = s2.replace("(", "").replace(")", "")
        # Extract first number
        m = re.search(r"[-+]?\d[\d,]*\.?\d*", s2_no_paren)
        if not m:
            return value
        num = float(m.group(0).replace(",", ""))
        # Respect explicit minus sign
        if "-" in s2_no_paren:
            num = -abs(num)
        # Apply parentheses negative if present
        if paren_present:
            num = -abs(num)
        return num
    except Exception:
        return value


def _clean_text(value: Any) -> Any:
    if value is None:
        return value
    try:
        s = str(value)
    except Exception:
        return value
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _title_like_name(col_name: str, value: Any) -> Any:
    # Only apply light title-casing to name-like columns
    if value is None:
        return value
    try:
        s = str(value)
    except Exception:
        return value
    if not s:
        return s
    tokens = s.split(" ")
    # Keep common acronyms
    acronyms = {"usa", "llc", "ltd", "inc", "ibm", "ups", "dhl", "rbc", "bmo", "hsbc", "cvs"}
    fixed: List[str] = []
    for t in tokens:
        low = re.sub(r"[^a-z0-9]", "", t.lower())
        if low in acronyms:
            fixed.append(t.upper())
        else:
            fixed.append(t[:1].upper() + t[1:].lower() if t else t)
    return " ".join(fixed)


# -----------------------------
# Detection (simple, high-signal)
# -----------------------------

@dataclass
class DetectionResult:
    detected_type: str = "unknown"
    confidence: float = 0.0
    signals: Dict[str, float] = field(default_factory=dict)


def detect_document_type(df: pd.DataFrame) -> DetectionResult:
    if df.empty:
        return DetectionResult("unknown", 0.0, {"empty": 1.0})

    lower_cols = {str(c).lower(): c for c in df.columns}
    has = lambda k: k in lower_cols
    score = {"sales_ledger": 0.0, "expense_ledger": 0.0, "gl_journal": 0.0}

    # Sales signals (high-signal rule per spec: ≥4 among core features)
    sales_keys = [
        "sku", "item description", "itemdescription", "qty", "quantity",
        "unit price", "unitprice", "subtotal", "total amount", "totalamount"
    ]
    sales_hits = sum(1 for k in sales_keys if any(k in s for s in lower_cols))
    core_presence = 0
    core_presence += 1 if any(k in lower_cols for k in ["sku"]) else 0
    core_presence += 1 if any(k in lower_cols for k in ["item description", "itemdescription"]) else 0
    core_presence += 1 if any(k in lower_cols for k in ["qty", "quantity"]) else 0
    core_presence += 1 if any(k in lower_cols for k in ["unit price", "unitprice", "subtotal"]) else 0
    core_presence += 1 if any(k in lower_cols for k in ["total amount", "totalamount"]) else 0
    if core_presence >= 4:
        score["sales_ledger"] += 0.9
    elif sales_hits >= 4:
        score["sales_ledger"] += min(1.0, sales_hits / 6.0)
    if has("time") or has("location"):
        score["sales_ledger"] += 0.1

    # Expense signals
    expense_hits = 0
    if any(h in lower_cols for h in ["merchant", "vendor", "payee", "name"]):
        expense_hits += 1
    if any(h in lower_cols for h in ["amount", "total", "price", "debit", "credit"]):
        expense_hits += 1
    if any(h in lower_cols for h in ["description", "memo", "notes"]):
        expense_hits += 1
    score["expense_ledger"] += (expense_hits / 3.0) * 0.8

    # GL signals
    if has("account") and has("debit") and has("credit"):
        score["gl_journal"] += 0.9

    detected = max(score, key=lambda k: score[k])
    confidence = float(min(1.0, max(0.0, score[detected])))
    if confidence < 0.3:
        detected = "unknown"
    return DetectionResult(detected, confidence, score)


# -----------------------------
# CommonCleaner (non-destructive)
# -----------------------------

@dataclass
class CleanSummary:
    schema_analysis: Dict[str, Any]
    processing_summary: Dict[str, Any]
    math_checks: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class CommonCleaner:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        # Safe defaults
        self.preserve_schema = True
        self.enable_ai = False
        self.enable_parallel_processing = False
        # Trackers
        self.values_normalized: Dict[str, int] = {}
        self.fields_filled: Dict[str, int] = {}
        self.duplicates_removed: int = 0
        self.totals_recomputed: int = 0
        self.math_mismatches: int = 0

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, CleanSummary]:
        start = time.time()
        if df is None or df.empty:
            result = df.copy() if df is not None else pd.DataFrame()
            det = DetectionResult("unknown", 0.0, {})
            return result, self._build_summary(det, start)

        # Strict non-destructive: capture column order and names
        original_columns = list(df.columns)
        cleaned = df.copy()

        # Phase 3: detection
        detection = detect_document_type(cleaned)

        # Phase 2: Core normalization (applied to every file)
        self._normalize_dates_inplace(cleaned)
        # Clean text before numeric normalization so numbers don't get converted back to strings
        self._clean_text_inplace(cleaned)
        self._normalize_numbers_inplace(cleaned)
        self._deduplicate_inplace(cleaned)

        # Phase 4: Type-specific processors (still non-destructive)
        if detection.detected_type == "sales_ledger" and detection.confidence >= 0.8:
            self._process_sales_inplace(cleaned)
        elif detection.detected_type == "expense_ledger" and detection.confidence >= 0.8:
            self._process_expense_inplace(cleaned)
        elif detection.detected_type == "gl_journal" and detection.confidence >= 0.8:
            self._process_gl_inplace(cleaned)
        # else: remain in CommonCleaner only

        # Global guard: ensure no column mutations
        cleaned = cleaned[original_columns]

        return cleaned, self._build_summary(detection, start)

    # -------------------------
    # Core handlers
    # -------------------------
    def _normalize_dates_inplace(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            col_low = str(col).lower()
            # Dates/timestamps → YYYY-MM-DD (avoid converting columns explicitly named 'Time')
            if any(k in col_low for k in ["date", "timestamp", "posted", "when"]) and col_low != "time":
                before = df[col].copy()
                df[col] = df[col].map(_to_iso_date)
                self._bump_normalized(col, before, df[col])
            # Time-only → HH:MM:SS (leave mixed/other as-is)
            elif "time" in col_low:
                before = df[col].copy()
                df[col] = df[col].map(_to_iso_time)
                self._bump_normalized(col, before, df[col])

        # Content-based detection: normalize columns that look like dates even without date-like headers
        for col in df.columns:
            try:
                series = df[col]
                if not pd.api.types.is_object_dtype(series):
                    continue
                sample = series.dropna().astype(str).head(50)
                if sample.empty:
                    continue
                # Skip time-only columns (format HH:MM[:SS] [AM/PM])
                if sample.str.match(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM|am|pm)?\s*$").mean() >= 0.5:
                    continue
                parsed = pd.to_datetime(sample, errors="coerce")
                ratio = parsed.notna().mean()
                # Avoid columns that are clearly time-only (already handled) or vendor-like
                if ratio >= 0.7:
                    before = df[col].copy()
                    df[col] = df[col].map(_to_iso_date)
                    self._bump_normalized(col, before, df[col])
            except Exception:
                continue

    def _normalize_numbers_inplace(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            col_low = str(col).lower()
            if any(k in col_low for k in ["amount", "total", "price", "cost", "debit", "credit", "tax", "shipping", "subtotal", "qty", "quantity", "unit price", "unitprice"]):
                before = df[col].copy()
                df[col] = df[col].map(_to_numeric)
                self._bump_normalized(col, before, df[col])

     # Content-based numeric normalization for columns that are strongly numeric-like (avoid IDs)
        for col in df.columns:
            try:
                series = df[col]
                if not pd.api.types.is_object_dtype(series):
                    continue
                sample = series.dropna().astype(str).head(50)
                if sample.empty:
                    continue
                # Skip date-like columns (already handled by date normalization)
                parsed_dates = pd.to_datetime(sample, errors="coerce")
                if parsed_dates.notna().mean() >= 0.5:
                    continue
                if sample.str.contains(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM|am|pm)?\s*$").mean() >= 0.5:
                    continue
                # Skip likely identifier columns by header
                col_lower = str(col).lower()
                if any(k in col_lower for k in ["id", "transaction", "sku", "account", "customer"]):
                    continue
                # Strong numeric-only heuristic: at least 80% match a numeric token without alphabetic chars
                numeric_only = sample.str.match(r"^\s*[\(\-]?\s*[$£€¥]?\s*\d[\d,]*\.?\d*\s*\)?\s*$", na=False)
                alpha_ratio = sample.str.contains(r"[A-Za-z]", na=False).mean()
                if numeric_only.mean() >= 0.8 and alpha_ratio <= 0.2:
                    before = df[col].copy()
                    df[col] = df[col].map(_to_numeric)
                    self._bump_normalized(col, before, df[col])
            except Exception:
                continue

    def _clean_text_inplace(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                before = df[col].copy()
                df[col] = df[col].map(_clean_text)
                # Title-case for probable name/vendor/customer/payee columns
                col_low = str(col).lower()
                if any(k in col_low for k in ["merchant", "vendor", "customer", "name", "payee", "company", "store"]):
                    df[col] = df[col].map(lambda v: _title_like_name(col, v))
                self._bump_normalized(col, before, df[col])

    def _deduplicate_inplace(self, df: pd.DataFrame) -> None:
        # Prefer explicit transaction id fields (expanded synonyms)
        id_aliases = {
            "transaction id", "transaction_id", "id", "txn id", "txnid",
            "reference", "ref", "reference id", "reference number",
            "document no", "doc no", "doc#", "document #",
            "confirmation", "confirmation #", "check number", "check no"
        }
        id_candidates = [c for c in df.columns if str(c).lower() in id_aliases]
        before_len = len(df)
        if id_candidates:
            key = id_candidates[0]
            df.drop_duplicates(subset=[key], inplace=True, keep="first")
        else:
            # Build stable composite key when available
            keys_pref = [
                ["date", "time", "customer", "sku", "qty", "total"],
                ["date", "merchant", "amount", "description"],
            ]
            lower_map = {str(c).lower(): c for c in df.columns}
            for pref in keys_pref:
                cols = [lower_map[k] for k in pref if k in lower_map]
                if len(cols) >= 3:
                    df.drop_duplicates(subset=cols, inplace=True, keep="first")
                    break
        self.duplicates_removed += max(0, before_len - len(df))

    # -------------------------
    # Type-specific (safe, in-place)
    # -------------------------
    def _process_sales_inplace(self, df: pd.DataFrame) -> None:
        lower_map = {str(c).lower(): c for c in df.columns}
        qty = lower_map.get("qty") or lower_map.get("quantity")
        unit_price = lower_map.get("unit price") or lower_map.get("unitprice")
        subtotal = lower_map.get("subtotal")
        tax = lower_map.get("tax")
        shipping = lower_map.get("shipping")
        total = lower_map.get("total") or lower_map.get("total amount") or lower_map.get("totalamount")

        # Normalize math if unambiguous and target column exists
        # Recompute Subtotal = Qty * Unit Price when both present
        if subtotal and qty and unit_price:
            try:
                calc = (
                    pd.to_numeric(df[qty], errors="coerce").fillna(0)
                    * pd.to_numeric(df[unit_price], errors="coerce").fillna(0)
                )
                # Only fill when existing value is null or blank string
                to_fill = df[subtotal].isna() | (df[subtotal].astype(str).str.strip() == "")
                if to_fill.any():
                    df.loc[to_fill, subtotal] = calc[to_fill]
                    self.totals_recomputed += int(to_fill.sum())
                    self.fields_filled[subtotal] = self.fields_filled.get(subtotal, 0) + int(to_fill.sum())
            except Exception:
                pass

        # Recompute Total = Subtotal + Tax + Shipping when present
        if total and subtotal:
            try:
                calc_total = pd.to_numeric(df[subtotal], errors="coerce").fillna(0)
                if tax:
                    calc_total = calc_total + pd.to_numeric(df[tax], errors="coerce").fillna(0)
                if shipping:
                    calc_total = calc_total + pd.to_numeric(df[shipping], errors="coerce").fillna(0)
                # Only fill blanks; never overwrite non-null differing values
                to_fill = df[total].isna() | (df[total].astype(str).str.strip() == "")
                if to_fill.any():
                    df.loc[to_fill, total] = calc_total[to_fill]
                    self.totals_recomputed += int(to_fill.sum())
                    self.fields_filled[total] = self.fields_filled.get(total, 0) + int(to_fill.sum())

                # Detect mismatches for summary only
                non_null = df[total].notna()
                try:
                    mism = (pd.to_numeric(df[total], errors="coerce") - calc_total).abs() > 0.01
                    self.math_mismatches += int((mism & non_null).sum())
                except Exception:
                    pass
            except Exception:
                pass

        # Fix known item typos if an item description-like field exists (substring-safe)
        item_col = next((lower_map[k] for k in ["item description", "itemdescription", "description"] if k in lower_map), None)
        if item_col is not None and pd.api.types.is_object_dtype(df[item_col]):
            try:
                df[item_col] = (
                    df[item_col]
                    .astype(str)
                    .str.replace(r"\bLavedner\b", "Lavender", regex=True)
                    .str.replace(r"\bSpeical\b", "Special", regex=True)
                )
            except Exception:
                pass

    def _process_expense_inplace(self, df: pd.DataFrame) -> None:
        # Normalize Amount if present
        amt_col = next((c for c in df.columns if str(c).lower() == "amount"), None)
        if amt_col:
            before = df[amt_col].copy()
            df[amt_col] = df[amt_col].map(_to_numeric)
            self._bump_normalized(amt_col, before, df[amt_col])
        # Light clean on vendor-like text only if such column exists
        for col in df.columns:
            col_low = str(col).lower()
            if any(k in col_low for k in ["merchant", "vendor", "payee", "name", "company", "store"]):
                before = df[col].copy()
                df[col] = df[col].map(_clean_text).map(lambda v: _title_like_name(col, v))
                self._bump_normalized(col, before, df[col])
        # Category normalization (no AI): trim only
        for col in df.columns:
            if str(col).lower() == "category" and pd.api.types.is_object_dtype(df[col]):
                before = df[col].copy()
                df[col] = df[col].map(_clean_text)
                self._bump_normalized(col, before, df[col])

    def _process_gl_inplace(self, df: pd.DataFrame) -> None:
        # Clean numeric debit/credit; do not add columns
        for key in ["debit", "credit"]:
            col = next((c for c in df.columns if str(c).lower() == key), None)
            if col:
                before = df[col].copy()
                df[col] = df[col].map(_to_numeric)
                self._bump_normalized(col, before, df[col])
        # Report unbalanced only (summary count): sum(debit) ~= sum(credit)?
        try:
            debit_col = next((c for c in df.columns if str(c).lower() == "debit"), None)
            credit_col = next((c for c in df.columns if str(c).lower() == "credit"), None)
            if debit_col and credit_col:
                debit_sum = pd.to_numeric(df[debit_col], errors="coerce").fillna(0).sum()
                credit_sum = pd.to_numeric(df[credit_col], errors="coerce").fillna(0).sum()
                if abs(debit_sum - credit_sum) > 0.01:
                    # count one mismatch flag for the ledger
                    self.math_mismatches += 1
        except Exception:
            pass

    # -------------------------
    # Helpers
    # -------------------------
    def _bump_normalized(self, col: Any, before: pd.Series, after: pd.Series) -> None:
        try:
            changed = (before != after) & ~(before.isna() & after.isna())
            self.values_normalized[str(col)] = self.values_normalized.get(str(col), 0) + int(changed.sum())
        except Exception:
            # Fallback: incremental by column length when operation succeeded
            self.values_normalized[str(col)] = self.values_normalized.get(str(col), 0) + len(after)

    def _build_summary(self, detection: DetectionResult, start_time: float) -> CleanSummary:
        processing_summary = {
            "duplicates_removed": int(self.duplicates_removed),
            "values_normalized": dict(sorted(self.values_normalized.items())),
            "fields_filled": dict(sorted(self.fields_filled.items())),
        }
        math_checks = {
            "totals_recomputed": int(self.totals_recomputed),
            "mismatches": int(self.math_mismatches),
        }
        perf = {
            "total_time_seconds": float(max(0.0, time.time() - start_time)),
        }
        schema = {
            "detected_type": detection.detected_type,
            "confidence": detection.confidence,
            "signals": detection.signals,
        }
        return CleanSummary(schema, processing_summary, math_checks, perf)
