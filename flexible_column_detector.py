#!/usr/bin/env python3

import pandas as pd
import re
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ColumnMapping:
    """Represents a detected column mapping with confidence"""
    amount_column: Optional[str] = None
    merchant_column: Optional[str] = None
    date_column: Optional[str] = None
    description_column: Optional[str] = None
    confidence: float = 0.0
    detection_method: str = "unknown"
    issues: List[str] = field(default_factory=list)

class FlexibleColumnDetector:
    """Smart column detector that handles any financial data format"""

    def detect_document_type(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect likely document type with a simple, explainable score.
        Returns: { 'detected_type': str, 'confidence': float, 'signals': Dict[str, Any], 'issues': List[str] }
        """
        result = {
            'detected_type': 'unknown',
            'confidence': 0.0,
            'signals': {},
            'issues': []
        }
        if df.empty:
            result['issues'].append('Empty DataFrame')
            return result

        cols = {str(c).lower(): c for c in df.columns}
        has = lambda k: k in cols
        score = {
            'sales_ledger': 0.0,
            'expense_ledger': 0.0,
            'gl_journal': 0.0
        }

        # Sales/POS indicators
        sales_keys = ['sku', 'itemdescription', 'unitprice', 'quantity', 'subtotal', 'totalamount', 'transactiontype']
        sales_hits = sum(1 for k in sales_keys if has(k))
        if sales_hits >= 4:
            score['sales_ledger'] += sales_hits / len(sales_keys) * 0.8
        if has('paymentmethod') or has('time') or has('location'):
            score['sales_ledger'] += 0.1

        # Expense ledger indicators
        expense_hits = 0
        if any(h in cols for h in ['merchant', 'vendor', 'payee', 'name']):
            expense_hits += 1
        if any(h in cols for h in ['amount', 'total', 'price', 'debit', 'credit']):
            expense_hits += 1
        if any(h in cols for h in ['description', 'memo', 'notes']):
            expense_hits += 1
        score['expense_ledger'] += (expense_hits / 3.0) * 0.8

        # GL/Journal indicators
        if has('account') and has('debit') and has('credit'):
            score['gl_journal'] += 0.9

        # Normalize and choose
        detected = max(score, key=lambda k: score[k])
        confidence = float(min(1.0, max(0.0, score[detected])))
        result['detected_type'] = detected if confidence >= 0.3 else 'unknown'
        result['confidence'] = confidence
        result['signals'] = score
        return result

    def normalize_to_dataframe(self, data: Any) -> pd.DataFrame:
        """
        Main entry point. Normalizes various input formats into a pandas DataFrame,
        including upfront validation and intelligent extraction for nested JSON.
        """
        # --- INTELLIGENT DATA EXTRACTION ---
        processed_data = None
        if isinstance(data, list) and all(isinstance(i, dict) for i in data):
            processed_data = data
        elif isinstance(data, dict):
            # If it's a dictionary of lists (columnar format), it's a valid structure
            if all(isinstance(v, list) for v in data.values()):
                processed_data = data
            else:
                # Otherwise, search for the first list of objects within the dict
                for key, value in data.items():
                    if isinstance(value, list) and all(isinstance(i, dict) for i in value):
                        logger.info(f"Found nested list of transactions under key: '{key}'")
                        processed_data = value
                        break
        
        if processed_data is None:
            logger.error(f"Invalid data format. Expected a list of records, a dictionary of lists, or a nested structure, but got {type(data)}.")
            return pd.DataFrame()

        try:
            df = pd.DataFrame(processed_data)
            if df.empty:
                logger.warning("Input data was valid but resulted in an empty DataFrame.")
            return df
        except Exception as e:
            logger.error(f"Failed to create DataFrame from input data: {e}", exc_info=True)
            return pd.DataFrame()

    def __init__(self):
        self.amount_patterns = ['amount', 'cost', 'price', 'total', 'debit', 'credit', 'value', 'payment']
        # Include common cleaned/standardized vendor fields from upstream pipelines
        self.merchant_patterns = [
            'standardized_vendor', 'clean vendor', 'clean_vendor',
            'merchant', 'vendor', 'store', 'business', 'payee', 'description', 'name'
        ]
        self.date_patterns = ['date', 'time', 'timestamp', 'when', 'posted']
        self.description_patterns = ['description', 'desc', 'note', 'memo', 'details']
        self.category_patterns = ['category', 'type', 'classification', 'group']

    def detect_columns(self, df: pd.DataFrame) -> ColumnMapping:
        """Main method to detect column mappings using multiple strategies"""
        if df.empty:
            return ColumnMapping(issues=["Input DataFrame is empty."])

        strategies = [
            self._detect_by_headers,
            self._detect_by_content_analysis
        ]
        
        best_mapping = ColumnMapping()
        
        for strategy in strategies:
            mapping = strategy(df)
            if mapping.confidence > best_mapping.confidence:
                best_mapping = mapping
            if best_mapping.confidence >= 0.9:
                break
        
        return self._validate_and_enhance_mapping(df, best_mapping)

    # --- Utilities expected by tests ---
    def _clean_amount_value(self, value: Any) -> Optional[float]:
        """Clean a currency/amount string to a float. Supports $, €, £, ¥ and codes.
        Handles commas, spaces, accounting negatives with parentheses, and signs.
        Returns float or None if no numeric content.
        """
        try:
            if value is None:
                return None
            s = str(value).strip()
            if s == "":
                return None
            paren_present = ("(" in s and ")" in s)
            # Remove currency symbols
            s2 = re.sub(r"[\$£€¥₹₽¢]", "", s)
            # Remove parentheses for extraction
            s2 = s2.replace("(", "").replace(")", "")
            # Extract first numeric token
            m = re.search(r"[-+]?\d[\d,]*\.?\d*", s2)
            if not m:
                return None
            num = float(m.group(0).replace(",", ""))
            # Respect explicit minus sign
            if "-" in s2:
                num = -abs(num)
            # Apply accounting negative
            if paren_present:
                num = -abs(num)
            return num
        except Exception:
            return None

    def _smart_date_parsing(self, series: pd.Series) -> (bool, str):
        """Infer date format from a series. Returns (success, format_used_or_empty).
        Tries common formats and falls back to pandas parsing if confident.
        """
        if series is None or len(series) == 0:
            return False, ""
        sample = series.dropna().astype(str).head(50)
        if sample.empty:
            return False, ""
        common_formats = [
            "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
            "%b %d, %Y", "%B %d, %Y", "%m-%d-%Y", "%d-%m-%Y"
        ]
        for fmt in common_formats:
            try:
                parsed = pd.to_datetime(sample, format=fmt, errors='coerce')
                if parsed.notna().mean() >= 0.8:
                    return True, fmt
            except Exception:
                continue
        # Fallback: generic parsing, require high success rate
        parsed = pd.to_datetime(sample, errors='coerce')
        if parsed.notna().mean() >= 0.8:
            return True, "generic"
        return False, ""

    def get_column_suggestions(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Suggest likely columns for amount, merchant, date based on headers and content."""
        suggestions: Dict[str, List[str]] = {"amount": [], "merchant": [], "date": [], "description": []}
        lower_map = {str(c).lower(): c for c in df.columns}
        # Header-based suggestions
        for patt in self.amount_patterns:
            for low, orig in lower_map.items():
                if patt in low and orig not in suggestions["amount"]:
                    suggestions["amount"].append(orig)
        for patt in self.merchant_patterns:
            for low, orig in lower_map.items():
                if patt in low and orig not in suggestions["merchant"]:
                    suggestions["merchant"].append(orig)
        for patt in self.date_patterns:
            for low, orig in lower_map.items():
                if patt in low and orig not in suggestions["date"]:
                    suggestions["date"].append(orig)
        for patt in self.description_patterns:
            for low, orig in lower_map.items():
                if patt in low and orig not in suggestions["description"]:
                    suggestions["description"].append(orig)
        # Content-based fallback
        for col in df.columns:
            series = df[col].dropna().head(100)
            if series.empty:
                continue
            # Amount-like
            if pd.to_numeric(series, errors='coerce').notna().mean() > 0.7 and col not in suggestions["amount"]:
                suggestions["amount"].append(col)
            # Date-like
            ok, _ = self._smart_date_parsing(series)
            if ok and col not in suggestions["date"]:
                suggestions["date"].append(col)
        return suggestions

    def _detect_by_headers(self, df: pd.DataFrame) -> ColumnMapping:
        """Detect columns based on header keyword matching"""
        mapping = ColumnMapping(detection_method="header_analysis")
        columns = {col.lower().strip(): col for col in df.columns}
        
        detected_fields = {}
        patterns = {
            'amount_column': self.amount_patterns,
            'merchant_column': self.merchant_patterns,
            'date_column': self.date_patterns,
            'description_column': self.description_patterns
        }

        for field, pats in patterns.items():
            for col_lower, col_original in columns.items():
                if any(p in col_lower for p in pats):
                    detected_fields[field] = col_original
                    break # Simple first-match logic
        
        mapping.amount_column = detected_fields.get('amount_column')
        mapping.merchant_column = detected_fields.get('merchant_column')
        mapping.date_column = detected_fields.get('date_column')
        mapping.description_column = detected_fields.get('description_column')

        # Calculate confidence
        found_count = sum(1 for f in [mapping.amount_column, mapping.merchant_column, mapping.date_column] if f)
        mapping.confidence = (found_count / 3.0) * 0.9
        
        return mapping

    def _detect_by_content_analysis(self, df: pd.DataFrame) -> ColumnMapping:
        """Detect columns by analyzing content patterns"""
        mapping = ColumnMapping(detection_method="content_analysis")
        scores = {col: {'amount': 0, 'date': 0, 'merchant': 0} for col in df.columns}

        for col in df.columns:
            series = df[col].dropna().head(100) # Sample 100 rows for performance
            if series.empty:
                continue
            
            # Amount scoring
            numeric_series = pd.to_numeric(series, errors='coerce')
            numeric_ratio = numeric_series.notna().mean()
            if numeric_ratio > 0.7:
                scores[col]['amount'] = numeric_ratio

            # Date scoring
            try:
                date_series = pd.to_datetime(series, errors='coerce')
                date_ratio = date_series.notna().mean()
                if date_ratio > 0.7:
                    scores[col]['date'] = date_ratio
            except Exception:
                pass

            # Merchant scoring (simple string properties)
            if series.dtype == 'object':
                avg_len = series.astype(str).str.len().mean()
                if 3 < avg_len < 50:
                    scores[col]['merchant'] = 0.5
        
        # Find best column for each type
        best_amount = max(scores, key=lambda c: scores[c]['amount']) if any(s['amount'] > 0 for s in scores.values()) else None
        best_date = max(scores, key=lambda c: scores[c]['date']) if any(s['date'] > 0 for s in scores.values()) else None
        best_merchant = max(scores, key=lambda c: scores[c]['merchant']) if any(s['merchant'] > 0 for s in scores.values()) else None

        mapping.amount_column = best_amount
        mapping.date_column = best_date
        mapping.merchant_column = best_merchant

        found_count = sum(1 for f in [best_amount, best_date, best_merchant] if f)
        mapping.confidence = (found_count / 3.0) * 0.8
        
        return mapping

    def _validate_and_enhance_mapping(self, df: pd.DataFrame, mapping: ColumnMapping) -> ColumnMapping:
        """Validate and enhance the detected mapping"""
        if not mapping.amount_column:
            mapping.issues.append("Amount column could not be detected.")
            mapping.confidence *= 0.5
        
        if not mapping.merchant_column:
            # If no merchant, fallback to description or first object column
            if mapping.description_column:
                mapping.merchant_column = mapping.description_column
                mapping.issues.append("Using description as merchant column.")
            else:
                object_cols = df.select_dtypes(include=['object']).columns
                if len(object_cols) > 0:
                    mapping.merchant_column = object_cols[0]
                    mapping.issues.append(f"Fallback: Using '{object_cols[0]}' as merchant column.")
        
        return mapping

def standardize_data_format(df: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    """Converts a DataFrame to a standardized format based on the detected mapping."""
    standard_df = pd.DataFrame()
    
    if mapping.amount_column and mapping.amount_column in df.columns:
        detector = FlexibleColumnDetector()
        standard_df['amount'] = df[mapping.amount_column].map(detector._clean_amount_value).astype(float)
        standard_df['amount'] = standard_df['amount'].fillna(0)
    else:
        standard_df['amount'] = 0

    if mapping.merchant_column and mapping.merchant_column in df.columns:
        standard_df['merchant'] = df[mapping.merchant_column].astype(str).fillna('')
    else:
        standard_df['merchant'] = ''

    if mapping.date_column and mapping.date_column in df.columns:
        # Try smart format first
        detector = FlexibleColumnDetector()
        ok, fmt = detector._smart_date_parsing(df[mapping.date_column])
        if ok and fmt and fmt != "generic":
            standard_df['date'] = pd.to_datetime(df[mapping.date_column], format=fmt, errors='coerce')
        else:
            standard_df['date'] = pd.to_datetime(df[mapping.date_column], errors='coerce')
    else:
        standard_df['date'] = pd.NaT
        
    if mapping.description_column and mapping.description_column in df.columns:
        standard_df['description'] = df[mapping.description_column].astype(str).fillna('')
    elif 'merchant' in standard_df.columns:
        standard_df['description'] = standard_df['merchant'] # Fallback
    else:
        standard_df['description'] = ''
        
    return standard_df
