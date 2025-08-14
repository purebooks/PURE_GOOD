#!/usr/bin/env python3
"""
AI-Enhanced Financial Data Cleaner API v5.0
Advanced LLM Flow with Intelligent Processing
Production-ready Flask application for Cloud Run deployment
"""

import os
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import re

try:
    from llm_client_v2 import LLMClient
except ImportError:
    from llm_client import LLMClient
from production_cleaner_ai_v5 import AIEnhancedProductionCleanerV5
from common_cleaner import CommonCleaner
from llm_assistant import LLMAssistant
from flexible_column_detector import FlexibleColumnDetector
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- App Setup ---
app = Flask(__name__)
CORS(app)

# --- Configuration ---
APP_CONFIG = {
    'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
    # Safe Mode defaults: AI off by default
    'enable_ai': os.getenv('ENABLE_AI', 'false').lower() == 'true',
    'port': int(os.getenv('PORT', 8080)),
    'debug': os.getenv('FLASK_ENV') == 'development',
    'max_file_size_mb': 50,
    'version': '5.0.0'
}

# --- Normalization helpers ---
ALLOWED_CATEGORIES = set([
    'Software & Technology',
    'Meals & Entertainment',
    'Travel & Transportation',
    'Office Supplies & Equipment',
    'Professional Services',
    'Banking & Finance',
    'Utilities & Rent',
    'Marketing & Advertising',
    'Employee Benefits',
    'Insurance & Legal',
    'Other'
])

VENDOR_PREFIX_SUFFIX_PATTERN = re.compile(
    r"^(?:PAYPAL\s*\*|SQ\s*\*|TST\s*\*)|(?:\s*#\d+$)|(?:\s*\*STORE\s*\d+$)|(?:\s*\.COM$)|(?:\s*ONLINE$)",
    re.IGNORECASE
)

PRODUCT_KEYWORDS = [
    'candle', 'mug', 'ceramic', 'journal', 'notebook', 'stapler', 'paper',
    'cheese', 'cracker', 'snack', 'olive oil', 'gift card', 'ticket', 'workshop',
    'wholesale', 'inventory'
]

def vendor_title_case(value: str) -> str:
    if not value:
        return value
    tokens = value.split()
    acronyms = {
        'att': 'AT&T', 'cvs': 'CVS', 'ups': 'UPS', 'ibm': 'IBM', 'usps': 'USPS',
        'dhl': 'DHL', 'hsbc': 'HSBC', 'rbc': 'RBC', 'bmo': 'BMO', 'kfc': 'KFC',
        'ikea': 'IKEA', 'amd': 'AMD', 'nvidia': 'NVIDIA', 'aaa': 'AAA', 'usa': 'USA',
    }
    brand_single = {
        'paypal': 'PayPal', 'youtube': 'YouTube', 'icloud': 'iCloud', 'iphone': 'iPhone',
        'itunes': 'iTunes', 'ebay': 'eBay', 'airbnb': 'Airbnb'
    }
    def normalize_token(tok: str) -> str:
        key = re.sub(r'[^a-z0-9&]', '', tok.lower())
        if key in ('at&t','att'):
            return 'AT&T'
        key_simple = re.sub(r'[^a-z0-9]', '', tok.lower())
        if key_simple in acronyms:
            return acronyms[key_simple]
        return tok[:1].upper() + tok[1:].lower() if tok else tok
    cased = [normalize_token(t) for t in tokens]
    result = ' '.join(cased)
    if len(cased) == 1:
        low = result.lower()
        if low in brand_single:
            return brand_single[low]
        if low == 'mcdonalds':
            return "McDonald’s"
    else:
        joined = []
        for t in cased:
            low = t.lower()
            if low in brand_single:
                joined.append(brand_single[low])
            elif low == 'mcdonalds':
                joined.append("McDonald’s")
            else:
                joined.append(t)
        result = ' '.join(joined)
    return result

def normalize_category(value: Any) -> str:
    try:
        s = '' if value is None else str(value).strip()
    except Exception:
        s = ''
    if not s or s.lower() == 'none':
        return 'Other'
    # Fast path exact match
    if s in ALLOWED_CATEGORIES:
        return s
    # Simple mapping heuristics
    low = s.lower()
    mapping = [
        # Core vendors/brands
        (['google','microsoft','adobe','aws','digitalocean','github','slack','zoom','netflix','spotify','dropbox','salesforce'], 'Software & Technology'),
        (['restaurant','food','cafe','coffee','pizza','burger','chipotle','starbucks','snack','cheese','cracker'], 'Meals & Entertainment'),
        (['uber','lyft','airlines','delta','united','southwest','hertz','budget','enterprise','shell','chevron','gas'], 'Travel & Transportation'),
        (['amazon','target','walmart','staples','office depot','home depot','best buy','costco','whole foods','safeway'], 'Office Supplies & Equipment'),
        (['consulting','services','professional','training','workshop','ticket'], 'Professional Services'),
        (['bank','chase','wells fargo','american express','visa','mastercard','paypal','gift card'], 'Banking & Finance'),
        (['verizon','at&t','comcast','internet','cable','electric','water','gas company'], 'Utilities & Rent'),
        (['meta','facebook','google ads','linkedin','twitter','marketing','advertising'], 'Marketing & Advertising'),
        (['benefits','insurance','health','dental','legal'], 'Insurance & Legal'),
        # Product nouns commonly seen in receipts
        (['candle','mug','ceramic','journal','notebook','paper','stapler','olive oil','wholesale','inventory'], 'Office Supplies & Equipment'),
    ]
    for keys, cat in mapping:
        if any(k in low for k in keys):
            return cat
    return 'Other'

def post_clean_vendor(value: Any) -> str:
    try:
        s = '' if value is None else str(value)
    except Exception:
        s = ''
    # Remove multiple patterns iteratively
    prev = None
    while prev != s:
        prev = s
        # Strip leading payment processor prefixes
        s = re.sub(r'^(PAYPAL\s*\*|SQ\s*\*|TST\s*\*)', '', s, flags=re.IGNORECASE).strip()
        # Strip trailing store ids / hashes / .COM / ONLINE
        s = re.sub(r'(\s*\*STORE\s*\d+\s*$)', '', s, flags=re.IGNORECASE).strip()
        s = re.sub(r'(\s*#\d+\s*$)', '', s, flags=re.IGNORECASE).strip()
        s = re.sub(r'(\s*\.COM\s*$)', '', s, flags=re.IGNORECASE).strip()
        s = re.sub(r'(\s*ONLINE\s*$)', '', s, flags=re.IGNORECASE).strip()
        # Strip corporate suffixes at end (Inc, LLC, Corp, Co, Ltd)
        s = re.sub(r'\b(inc|llc|corp|co|ltd)\.?\s*$', '', s, flags=re.IGNORECASE).strip()
    # Normalize whitespace
    s = ' '.join(s.split())
    # Title-case with acronym and brand preservation
    s = vendor_title_case(s)
    return s

def looks_like_product_name(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    if any(k in low for k in PRODUCT_KEYWORDS):
        return True
    # Heuristic: presence of " - " often separates product and variant
    if ' - ' in text:
        return True
    # Heuristic: title-cased multiword with no common company suffixes may be a product
    if re.search(r"\b(inc|llc|corp|co|ltd)\b", low):
        return False
    words = text.split()
    if 2 <= len(words) <= 5 and sum(1 for w in words if w and w[0].isupper()) >= len(words) - 1:
        return True
    return False

# --- Optional memo enrichment with external LLM (OpenAI GPT-5 via compat client) ---
def _infer_rule_memo(vendor: str, category: str) -> str:
    v = (vendor or '').lower()
    c = (category or '')
    if 'paypal' in v or 'stripe' in v or 'linkedin' in v or 'google' in v:
        return 'Subscription'
    if c == 'Travel & Transportation':
        return 'Travel'
    if c == 'Office Supplies & Equipment':
        return 'Office supplies'
    if c == 'Marketing & Advertising':
        return 'Marketing'
    return 'Business expense'

def _memo_needs_enrichment(text: str) -> bool:
    return not bool((text or '').strip())

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
def _gpt5_batch_enrich(memos: list, provider_cfg: dict) -> list:
    """Best-effort memo enrichment. Never raises; returns empty memos on any failure."""
    try:
        try:
            from openai import OpenAI
        except Exception as import_err:
            logger.warning(f"OpenAI SDK not available for memo enrichment: {import_err}")
            return [''] * len(memos)

        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            return [''] * len(memos)

        try:
            client = OpenAI(api_key=api_key)
        except Exception as init_err:
            logger.warning(f"OpenAI client init failed; skipping enrichment: {init_err}")
            return [''] * len(memos)

        # Build concise prompt
        lines = []
        for i, m in enumerate(memos, 1):
            vendor = m.get('vendor', '')
            category = m.get('category', '')
            notes = m.get('notes', '')
            amount = m.get('amount', 0)
            lines.append(f"{i}. vendor={vendor}; category={category}; amount=${amount:.2f}; notes={notes}")
        prompt = (
            "You generate concise, non-marketing memo labels (<= 6 words). "
            "Return ONLY a JSON array of strings; no extra text.\n" + "\n".join(lines)
        )

        try:
            resp = client.chat.completions.create(
                model='gpt-5-mini',
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            text = (resp.choices[0].message.content or '').strip()
            # Extract JSON array
            start = text.find('['); end = text.rfind(']') + 1
            if start != -1 and end > start:
                import json as _json
                arr = _json.loads(text[start:end])
                if isinstance(arr, list):
                    # Normalize to strings and cap length
                    return [str(x)[:64] for x in arr][:len(memos)]
        except Exception as call_err:
            logger.warning(f"OpenAI enrichment call failed; using rule memos: {call_err}")
            return [''] * len(memos)

    except Exception as unexpected:
        logger.warning(f"Unexpected memo enrichment error: {unexpected}")
    return [''] * len(memos)

# --- Global State ---
llm_client = None
start_time = datetime.utcnow()

# --- Helper Functions ---
def standardize_date(value):
    """
    Parse many date formats or timestamp-like objects into YYYY-MM-DD.
    Returns None if parsing fails.
    """
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        if value is None:
            return None
    try:
        parsed = pd.to_datetime(value, errors='coerce')
        if pd.notna(parsed):
            return parsed.strftime('%Y-%m-%d')
        return None
    except Exception:
        return None

def standardize_amount(amount_val):
    """
    Sanitizes an amount value to a standard float.
    Handles currency symbols, commas, parentheses for negatives, and other text.
    """
    if amount_val is None or pd.isna(amount_val):
        return 0.00
    
    s_amount = str(amount_val).strip()
    
    is_negative = False
    # Check for accounting-style negatives, e.g., (15.25)
    if s_amount.startswith('(') and s_amount.endswith(')'):
        is_negative = True
        s_amount = s_amount[1:-1]
        
    # Use regex to find the first valid number (int or float) in the string
    # This will strip out currency symbols, text like "Amount:", etc.
    import re
    match = re.search(r'[\d,]+\.?\d*', s_amount)
    
    if match:
        num_str = match.group(0).replace(',', '')
        try:
            amount = float(num_str)
            # If the original string had a negative sign, respect it
            if '-' in s_amount:
                amount = -abs(amount)
            # Apply negative sign if it was in accounting format
            elif is_negative:
                amount = -abs(amount)
            return amount
        except (ValueError, TypeError):
            return 0.00
            
    return 0.00

def get_llm_client() -> LLMClient:
    """Initializes and returns a singleton LLM client."""
    global llm_client
    if llm_client is None:
        use_mock = not APP_CONFIG['anthropic_api_key'] or not APP_CONFIG['enable_ai']
        if use_mock:
            logger.warning("Using mock AI client. Set ANTHROPIC_API_KEY for live AI.")
        llm_client = LLMClient(api_key=APP_CONFIG['anthropic_api_key'], use_mock=use_mock)
    return llm_client

def preprocess_and_standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    The core data purification and standardization pipeline.
    This runs BEFORE any AI processing.
    """
    # Find the most likely date and amount columns (prefer 'posted' first for bank exports)
    date_col = next((col for col in df.columns if col.lower() in ['posted', 'date', 'transaction date']), None)
    amount_col = next((col for col in df.columns if col.lower() in ['amount', 'price', 'cost', 'value', 'total']), None)
    
    # Standardize dates
    if date_col:
        df[date_col] = df[date_col].apply(standardize_date)
        df = df.rename(columns={date_col: 'Transaction Date'}) # Standardize name
    else:
        df['Transaction Date'] = None

    # Standardize amounts
    if amount_col:
        df[amount_col] = df[amount_col].apply(standardize_amount)
        df = df.rename(columns={amount_col: 'Amount'}) # Standardize name
    else:
        df['Amount'] = 0.00
        
    # --- Consolidate Description-like Columns (Moved from /process) ---
    description_cols = [col for col in df.columns if col.lower() in ['description', 'memo', 'notes']]
    if len(description_cols) > 1:
        df['Consolidated Description'] = df[description_cols].astype(str).agg(' | '.join, axis=1)
        df['Consolidated Description'] = df['Consolidated Description'].str.replace('nan', '').str.replace(r'\s*\|\s*\|*\s*', ' | ', regex=True).str.strip(' |')
        df = df.drop(columns=description_cols)
        df = df.rename(columns={'Consolidated Description': 'Description'})
    elif len(description_cols) == 1:
        df = df.rename(columns={description_cols[0]: 'Description'})

    # --- Phase 3 & 4: AI Imputation and "Needs Review" Flag ---
    # Prefer already-cleaned vendor-like columns first, then typical merchant fields
    preferred_vendor_candidates = [
        'standardized_vendor', 'clean vendor', 'clean_vendor',
        'merchant', 'vendor', 'store', 'business', 'payee', 'name'
    ]
    merchant_col = None
    for cand in preferred_vendor_candidates:
        for col in df.columns:
            if str(col).lower() == cand:
                merchant_col = col
                break
        if merchant_col:
            break
    if not merchant_col:
        # Fallback: substring match on common vendor tokens
        merchant_col = next((col for col in df.columns if any(k in str(col).lower() for k in ['merchant','vendor','store','business','payee','name'])), None)

    if merchant_col:
        df = df.rename(columns={merchant_col: 'Merchant'})
    else:
        df = df.assign(Merchant=None)

    df['Needs Review'] = False
    df['Suggestions'] = [[] for _ in range(len(df))]

    missing_merchant_mask = df['Merchant'].isnull() | (df['Merchant'] == '')
    if missing_merchant_mask.any():
        logger.info(f"Found {missing_merchant_mask.sum()} rows with missing merchants. Attempting AI imputation.")
        imputation_client = get_llm_client()
        
        for index, row in df[missing_merchant_mask].iterrows():
            description = row.get('Description', '')
            amount = row.get('Amount', 0.0)
            
            if description:
                try:
                    suggestions = imputation_client.suggest_vendors_from_description(description, amount)
                    if suggestions:
                        df.loc[index, 'Suggestions'] = [suggestions] # Nest suggestions in a list for the cell
                        df.loc[index, 'Needs Review'] = True
                        df.loc[index, 'Merchant'] = '[Vendor Missing]' # Keep placeholder for now
                        logger.info(f"AI suggested vendors {suggestions} for row {index}")
                    else:
                        df.loc[index, 'Merchant'] = '[Vendor Missing]'
                        df.loc[index, 'Needs Review'] = True
                except Exception as e:
                    logger.error(f"AI imputation failed for row {index}: {e}")
                    df.loc[index, 'Merchant'] = '[Vendor Missing]'
                    df.loc[index, 'Needs Review'] = True
            else:
                df.loc[index, 'Merchant'] = '[Vendor Missing]'
                df.loc[index, 'Needs Review'] = True
                
    # Final fallback for any remaining missing merchants
    df['Merchant'] = df['Merchant'].fillna('[Vendor Missing]')
    
    return df



def _is_unknown_vendor(merchant: str, amount: float = 0) -> bool:
    """Enhanced check for vendors that would benefit from AI processing."""
    if not merchant or not isinstance(merchant, str):
        return False
    
    merchant_lower = merchant.lower()
    
    # HIGH PRIORITY: Always use AI for high-value transactions
    if amount > 500:
        return True
    
    # HIGH PRIORITY: Complex payment processor patterns need AI parsing
    complex_processors = ["paypal *", "sq *", "stripe*", "tst*", "pp*", "venmo*"]
    if any(merchant.upper().startswith(proc.upper()) for proc in complex_processors):
        return True
    
    # HIGH PRIORITY: Cryptic/coded merchant names benefit from AI
    if any(char in merchant for char in ["*", "#", ".", "1234567890"]) and len(merchant) > 10:
        return True
    
    # Known vendor patterns (confident rule-based processing)
    known_patterns = [
        "google", "amazon", "netflix", "spotify", "apple", "microsoft", "adobe",
        "starbucks", "mcdonald", "chipotle", "subway", "pizza", "domino",
        "uber", "lyft", "shell", "chevron", "delta", "united", "southwest",
        "target", "walmart", "costco", "home depot", "best buy",
        "cvs", "walgreens", "whole foods", "safeway", "kroger",
        "bank", "chase", "wells fargo", "american express",
        "at&t", "verizon", "comcast", "hilton", "marriott", "airbnb"
    ]
    
    # Check if merchant contains any known patterns
    for pattern in known_patterns:
        if pattern in merchant_lower:
            return False
    
    # MEDIUM PRIORITY: Business entities that need context understanding
    business_keywords = ["corp", "inc", "llc", "ltd", "company", "consulting", 
                        "restaurant", "cafe", "store", "shop", "market", "solutions",
                        "services", "group", "enterprises", "technologies"]
    
    for keyword in business_keywords:
        if keyword in merchant_lower:
            return True  # AI can provide better context and categorization
    
    # MEDIUM PRIORITY: Subscription-like patterns
    if any(word in merchant_lower for word in ["subscription", "monthly", "annual", "renewal"]):
        return True
    
    # If no patterns match, likely needs AI processing
    return True

def _compute_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute quality metrics on the standardized display dataframe.
    Expects columns: 'Transaction Date', 'Amount', 'Clean Vendor', 'Category'.
    """
    try:
        total = max(1, len(df))
        # Date parsed (non-empty and parseable)
        dt_parsed = pd.to_datetime(df['Transaction Date'], errors='coerce') if 'Transaction Date' in df.columns else pd.Series([pd.NaT] * total)
        date_parse_rate = float((dt_parsed.notna()).sum()) / float(total)

        # Amount valid (not NaN)
        amt_series = pd.to_numeric(df['Amount'], errors='coerce') if 'Amount' in df.columns else pd.Series([float('nan')] * total)
        amount_valid_rate = float((amt_series.notna()).sum()) / float(total)

        # Vendor non-missing
        clean_vendor = df['Clean Vendor'] if 'Clean Vendor' in df.columns else pd.Series([''] * total)
        vendor_non_missing_rate = float((clean_vendor.astype(str).str.strip() != '[Vendor Missing]').sum()) / float(total)

        # Category non-Other and in allowed
        cat_series = df['Category'] if 'Category' in df.columns else pd.Series([''] * total)
        in_allowed = cat_series.astype(str).apply(lambda c: c in ALLOWED_CATEGORIES)
        non_other = cat_series.astype(str) != 'Other'
        category_non_other_rate = float((in_allowed & non_other).sum()) / float(total)

        # Duplicate rate (by Date+Amount+Vendor)
        dup_subset = pd.DataFrame({
            'd': df['Transaction Date'] if 'Transaction Date' in df.columns else '',
            'a': df['Amount'] if 'Amount' in df.columns else 0.0,
            'v': df['Clean Vendor'] if 'Clean Vendor' in df.columns else ''
        })
        dup_mask = dup_subset.duplicated(keep='first')
        duplicate_rate = float(dup_mask.sum()) / float(total)

        return {
            'rows': int(total),
            'date_parse_rate': date_parse_rate,
            'amount_valid_rate': amount_valid_rate,
            'vendor_non_missing_rate': vendor_non_missing_rate,
            'category_non_other_rate': category_non_other_rate,
            'duplicate_rate': duplicate_rate,
        }
    except Exception as e:
        logger.warning(f"Quality metrics computation failed: {e}")
        return {
            'rows': len(df) if isinstance(df, pd.DataFrame) else 0,
            'date_parse_rate': 0.0,
            'amount_valid_rate': 0.0,
            'vendor_non_missing_rate': 0.0,
            'category_non_other_rate': 0.0,
            'duplicate_rate': 0.0,
        }

def safe_dataframe_to_json(df: pd.DataFrame) -> list:
    """Converts a DataFrame to a list of records, safely handling NaT and NaN."""
    import numpy as np
    import math
    import ast
    from collections import OrderedDict
    
    df_copy = df.copy()
    
    # Handle datetime columns
    for col in df_copy.select_dtypes(include=['datetime64[ns]']).columns:
        df_copy[col] = df_copy[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)
    
    # Replace all NaN values with None (which becomes null in JSON)
    df_copy = df_copy.replace({np.nan: None})
    
    # Use pandas to_dict with orient='records' to preserve order, then reorder manually
    records = []
    for _, row in df_copy.iterrows():
        # Create record in exact order we want
        record = {
            'Transaction Date': row.get('Transaction Date', ''),
            'Amount': row.get('Amount', ''),
            'Clean Vendor': row.get('Clean Vendor', ''),
            'Category': row.get('Category', ''),
            'Description/Memo': row.get('Description/Memo', ''),
            'Needs Review': row.get('Needs Review', False),
            'Suggestions': row.get('Suggestions', [])
        }
        
        # Extract vendor suggestions from explanation if present
        explanation = row.get('vendor_explanation', '') or row.get('explanation', '')
        if explanation and 'AI vendor suggestions:' in explanation:
            try:
                suggestions_str = explanation.split('AI vendor suggestions:')[1].strip()
                suggestions = ast.literal_eval(suggestions_str) if suggestions_str.startswith('[') else []
                if isinstance(suggestions, list):
                    record['Vendor Suggestions'] = suggestions
            except Exception:
                record['Vendor Suggestions'] = []
        # Clean up NaN values
        for key, value in record.items():
            if pd.isna(value) or (isinstance(value, float) and math.isnan(value)):
                record[key] = None
            elif value == 'nan' or str(value).strip() == '':
                if key in ['Clean Vendor', 'Category']:
                    record[key] = f'[{key.replace("Clean ", "").replace("/Memo", "")} Missing]'
                else:
                    record[key] = None
        
        records.append(record)
    
    return records

def create_demo_data():
    """Creates sample data for demo endpoint."""
    return {
        "merchant": [
            "PAYPAL*DIGITALOCEAN",
            "SQ *COFFEE SHOP NYC", 
            "UBER EATS DEC15",
            "AMAZON.COM*AMZN.COM/BILL",
            "NETFLIX.COM"
        ],
        "amount": [50.00, 4.50, 23.75, 12.99, 15.99],
        "description": [
            "DigitalOcean hosting payment",
            "Coffee purchase at local shop",
            "Food delivery service",
            "Amazon Prime subscription",
            "Netflix streaming service"
        ]
    }

# --- API Endpoints ---
@app.route('/health', methods=['GET'])
def health_check():
    """Confirms the API is running."""
    uptime = (datetime.utcnow() - start_time).total_seconds()
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.utcnow().isoformat(),
        'version': APP_CONFIG['version'],
        'uptime_seconds': uptime
    })

@app.route('/config', methods=['GET'])
def get_config():
    """Returns API configuration."""
    return jsonify({
        'enable_ai': APP_CONFIG['enable_ai'],
        'has_api_key': bool(APP_CONFIG['anthropic_api_key']),
        'max_file_size_mb': APP_CONFIG['max_file_size_mb'],
        'version': APP_CONFIG['version']
    })

@app.route('/demo', methods=['POST'])
def demo_endpoint():
    """Demo endpoint with sample data processing."""
    request_id = f"demo-{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    
    try:
        demo_data = create_demo_data()

        # Normalize input and run Safe Mode cleaner (non-destructive)
        detector = FlexibleColumnDetector()
        df = detector.normalize_to_dataframe(demo_data)
        cleaner = CommonCleaner()
        cleaned_df, summary = cleaner.clean(df)

        processing_time = time.time() - start_time

        cleaned_records = cleaned_df.replace({pd.NA: None, np.nan: None}).to_dict(orient='records')
        return jsonify({
            'cleaned_data': cleaned_records,
            'summary_report': {
                'schema_analysis': summary.schema_analysis,
                'processing_summary': summary.processing_summary,
                'math_checks': summary.math_checks,
                'performance_metrics': summary.performance_metrics,
            },
            'insights': {
                'ai_requests': 0,
                'ai_cost': 0.0,
                'processing_time': processing_time,
                'rows_processed': len(cleaned_df)
            },
            'processing_time': processing_time,
            'request_id': request_id,
        })

    except Exception as e:
        logger.error(f"[{request_id}] Demo processing error: {e}", exc_info=True)
        return jsonify({'error': 'Demo processing failed.', 'details': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Returns API statistics."""
    uptime = (datetime.utcnow() - start_time).total_seconds()
    return jsonify({
        'uptime_seconds': uptime,
        'version': APP_CONFIG['version'],
        'ai_enabled': APP_CONFIG['enable_ai'],
        'has_api_key': bool(APP_CONFIG['anthropic_api_key'])
    })

@app.route('/process', methods=['POST'])
def process_data():
    """Main data processing endpoint."""
    request_id = f"req-{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    
    try:
        # --- NEW DEBUG LOGGING ---
        raw_body = request.get_data(as_text=True)
        logger.info(f"[{request_id}] ----- RAW REQUEST BODY -----")
        logger.info(raw_body)
        logger.info(f"[{request_id}] --- END RAW REQUEST BODY ---")

        request_data = request.get_json()
        logger.info(f"[{request_id}] Parsed JSON type: {type(request_data)}")
        
        user_intent = request_data.get('user_intent', '')
        
        # Extract the actual data from the request
        data = request_data.get('data', request_data)
        logger.info(f"[{request_id}] Extracted 'data' block type: {type(data)}")
        
        # Safe Mode: defaults
        config_override = request_data.get('config', {})
        preserve_schema = bool(config_override.get('preserve_schema', True))
        enable_ai = bool(config_override.get('enable_ai', False))
        enable_parallel_processing = bool(config_override.get('enable_parallel_processing', False))
        
        # Use FlexibleColumnDetector to handle various input formats
        detector = FlexibleColumnDetector()
        df = detector.normalize_to_dataframe(data)
        detection = detector.detect_document_type(df)

        # ---------------------------------------------
        # Safe Mode short-circuit: CommonCleaner pathway
        # Always use CommonCleaner when preserving schema; ignore AI flags for core cleaning
        # ---------------------------------------------
        if preserve_schema:
            cleaner = CommonCleaner()
            cleaned_df, summary = cleaner.clean(df)

            llm_block = {}
            if bool(config_override.get('enable_ai', False)):
                # Default to Assist in preserve mode unless explicitly allowed to apply
                cfg = dict(config_override)
                if 'ai_mode' not in cfg:
                    cfg['ai_mode'] = 'assist'
                assistant = LLMAssistant(list(ALLOWED_CATEGORIES))
                suggestions, applied, stats = assistant.enhance(cleaned_df, cfg)
                llm_block = {
                    'rows_considered': len(suggestions),
                    'rows_applied': applied,
                    'stats': stats,
                }

            cleaned_records = cleaned_df.replace({pd.NA: None, np.nan: None}).to_dict(orient='records')
            processing_time = time.time() - start_time
            return jsonify({
                'cleaned_data': cleaned_records,
                'summary_report': {
                    'schema_analysis': summary.schema_analysis,
                    'processing_summary': summary.processing_summary,
                    'math_checks': summary.math_checks,
                    'performance_metrics': summary.performance_metrics,
                    'llm': llm_block,
                },
                'insights': {
                    'ai_requests': llm_block.get('stats', {}).get('tracker', {}).get('calls', 0) if llm_block else 0,
                    'ai_cost': llm_block.get('stats', {}).get('tracker', {}).get('total_cost', 0.0) if llm_block else 0.0,
                    'processing_time': processing_time,
                    'rows_processed': len(cleaned_df),
                },
                'processing_time': processing_time,
                'request_id': request_id,
            })
        
        # --- Legacy path retained for non-preserve runs ---
        if not preserve_schema:
            df = preprocess_and_standardize_data(df)
            df['source_row_id'] = range(len(df))

        if df.empty:
            return jsonify({'error': 'No valid transaction data found.'}), 400

        logger.info(f"[{request_id}] Processing {len(df)} rows. Intent: '{user_intent or 'None'}'")

        # config_override already parsed above

        # Auto-route features based on detected type when preserving schema
        if preserve_schema:
            if detection.get('detected_type') == 'sales_ledger':
                # Disable AI category/vendor for sales; we won't add columns
                config_override['enable_ai'] = False
                config_override['ai_vendor_enabled'] = False
                config_override['ai_category_enabled'] = False
            elif detection.get('detected_type') == 'gl_journal':
                config_override['enable_ai'] = False
                config_override['ai_vendor_enabled'] = False
                config_override['ai_category_enabled'] = False
        use_real_llm = config_override.get('use_real_llm', True) or not config_override.get('use_mock', False)
        
        if use_real_llm and APP_CONFIG['anthropic_api_key']:
            # Smart hybrid approach: pre-analyze vendors
            unknown_vendors = []
            for row in df.itertuples():
                merchant = str(getattr(row, 'Merchant', '') or getattr(row, 'merchant', ''))
                amount = abs(float(getattr(row, 'Amount', 0) or getattr(row, 'amount', 0) or 0))
                if merchant and _is_unknown_vendor(merchant, amount):
                    unknown_vendors.append(merchant)
            
            if unknown_vendors:
                logger.info(f"[{request_id}] Found {len(unknown_vendors)} unknown vendors, using REAL LLM for enhancement")
                client = LLMClient(api_key=APP_CONFIG['anthropic_api_key'], use_mock=False)
            else:
                logger.info(f"[{request_id}] All vendors recognized, using efficient Python rules")
                client = get_llm_client()
        else:
            # Use default (mock) client
            client = get_llm_client()
            
        # Pass through request-level config so flags like force_llm_for_testing/use_mock take effect
        cleaner = AIEnhancedProductionCleanerV5(df, config=config_override, llm_client=client, user_intent=user_intent)
        cleaned_df, report = cleaner.process_data()

        # Attach schema detection summary
        if 'summary_report' in report:
            report['summary_report']['schema_analysis'] = {
                'detected_type': detection.get('detected_type', 'unknown'),
                'confidence': detection.get('confidence', 0.0),
                'signals': detection.get('signals', {}),
            }

        processing_time = time.time() - start_time
        logger.info(f"[{request_id}] Processing finished in {processing_time:.2f}s")
        
        # --- DEBUG: Log DataFrame shapes ---
        logger.info(f"[{request_id}] Input DataFrame shape: {df.shape}")
        logger.info(f"[{request_id}] Output DataFrame shape: {cleaned_df.shape}")
        logger.info(f"[{request_id}] Output DataFrame columns: {list(cleaned_df.columns)}")
        if len(cleaned_df) <= 5:
            logger.info(f"[{request_id}] Output DataFrame preview:\n{cleaned_df}")
        else:
            logger.info(f"[{request_id}] Output DataFrame head:\n{cleaned_df.head()}")

        # Extract insights from the report (normalized fields)
        summary = report.get('summary_report', {})
        processing_summary = summary.get('processing_summary', {})
        cost_analysis = summary.get('cost_analysis', {})
        
        # Create vendor transformations summary
        vendor_transformations = []
        if 'vendor_standardization' in report.get('summary_report', {}):
            for original, standardized in report['summary_report']['vendor_standardization'].items():
                vendor_transformations.append(f"{original} → {standardized}")

        # Clean up the output for professional display
        display_df = cleaned_df.copy()
        
        # Create the exact columns we want in the exact order requested
        final_df = pd.DataFrame()

        # If preserving schema, don't change headers or add/remove columns; only sanitize in-place when possible
        if preserve_schema:
            # Return cleaned_df as-is (values already improved in-place by cleaner); no reformatting
            cleaned_records = cleaned_df.replace({pd.NA: None}).to_dict(orient='records')
            return jsonify({
                'cleaned_data': cleaned_records,
                'summary_report': report.get('summary_report', {}),
                'insights': {
                    'ai_requests': report.get('summary_report', {}).get('processing_summary', {}).get('llm_calls', 0),
                    'ai_cost': report.get('summary_report', {}).get('cost_analysis', {}).get('total_cost', 0.0),
                    'processing_time': processing_time,
                    'rows_processed': len(cleaned_df),
                    'vendor_transformations': []
                },
                'processing_time': processing_time,
                'request_id': request_id,
            })
        
        # Column 1: Transaction Date - find, standardize, and assign (prefer 'posted' first)
        date_col = next((col for col in display_df.columns if col.lower() in ['posted', 'date', 'transaction date']), None)
        if date_col:
            final_df['Transaction Date'] = display_df[date_col].apply(standardize_date)
        else:
            final_df['Transaction Date'] = '[Date Missing]'

        # Fallback: if Transaction Date failed to parse, try alternate headers
        try:
            mask_blank = (
                final_df['Transaction Date'].isna() |
                (final_df['Transaction Date'].astype(str).str.strip() == '') |
                (final_df['Transaction Date'].astype(str).str.strip() == '[Date Missing]')
            )
            if mask_blank.any():
                for alt in ['Transaction Date', 'Posted', 'Date']:
                    if alt in display_df.columns:
                        parsed = display_df[alt].apply(standardize_date)
                        final_df.loc[mask_blank, 'Transaction Date'] = final_df.loc[mask_blank, 'Transaction Date'].combine_first(parsed[mask_blank])
                        mask_blank = final_df['Transaction Date'].isna() | (final_df['Transaction Date'].astype(str).str.strip() == '')
                        if not mask_blank.any():
                            break
        except Exception:
            pass

            
                # Column 2: Amount - find, standardize, and assign
        amount_col = next((col for col in display_df.columns if col.lower() in ['amount', 'price', 'cost', 'value', 'total']), None)
        if amount_col:
            final_df['Amount'] = display_df[amount_col].apply(standardize_amount)
        else:
            final_df['Amount'] = 0.00

            
        # Column 3: Clean Vendor - check multiple possible names
        vendor_col = None
        for col in ['standardized_vendor', 'Merchant', 'merchant', 'vendor', 'store', 'business']:
            if col in display_df.columns:
                vendor_col = col
                break
                
        if vendor_col:
            final_df['Clean Vendor'] = display_df[vendor_col]
        else:
            final_df['Clean Vendor'] = ''
            
        # Column 4: Category  
        if 'category' in display_df.columns:
            final_df['Category'] = display_df['category']
        else:
            final_df['Category'] = ''
            
        # Column 5: Description/Memo - check multiple possible names and combine if needed
        description_col = None
        memo_col = None
        
        for col in ['Notes', 'notes', 'description', 'Description', 'desc', 'details']:
            if col in display_df.columns:
                description_col = col
                break
                
        for col in ['memo', 'Memo', 'comment', 'Comment']:
            if col in display_df.columns:
                memo_col = col
                break
        
        if description_col and memo_col:
            # Combine description and memo
            final_df['Description/Memo'] = (display_df[description_col].fillna('').astype(str) + 
                                          ' | ' + display_df[memo_col].fillna('').astype(str)).str.strip(' | ')
        elif description_col:
            final_df['Description/Memo'] = display_df[description_col].fillna('')
        elif memo_col:
            final_df['Description/Memo'] = display_df[memo_col].fillna('')
        else:
            final_df['Description/Memo'] = ''
        
        # Capture pre-normalized vendor to inform category decision
        pre_normalized_vendor = final_df['Clean Vendor'].copy()

        # Normalize vendor for output (no processor prefixes); demote product-like strings to missing vendor
        def _normalize_vendor_out(x: Any) -> str:
            s = post_clean_vendor(x)
            if looks_like_product_name(s):
                return '[Vendor Missing]'
            return s or '[Vendor Missing]'
        final_df['Clean Vendor'] = final_df['Clean Vendor'].apply(_normalize_vendor_out)

        # Prefer model/rule category from cleaned_df when valid; otherwise fallback to heuristic normalization
        def choose_category(idx: int) -> str:
            try:
                raw_cat = cleaned_df.at[idx, 'category'] if 'category' in cleaned_df.columns else ''
            except Exception:
                raw_cat = ''
            raw_cat_str = '' if raw_cat is None else str(raw_cat).strip()
            # If current category is allowed but is 'Other' while vendor looks like a product, override
            if raw_cat_str in ALLOWED_CATEGORIES:
                if raw_cat_str == 'Other':
                    vend_pre = str(pre_normalized_vendor.at[idx] or '')
                    if looks_like_product_name(vend_pre):
                        return 'Office Supplies & Equipment'
                return raw_cat_str
            base = final_df.at[idx, 'Category'] if 'Category' in final_df.columns else ''
            # If vendor looks like a product, prefer Office Supplies & Equipment
            vend_pre = str(pre_normalized_vendor.at[idx] or '')
            if looks_like_product_name(vend_pre):
                return 'Office Supplies & Equipment'
            # Otherwise use heuristic normalization
            return normalize_category(base)

        final_df['Category'] = [choose_category(i) for i in final_df.index]
        final_df['Transaction Date'] = final_df['Transaction Date'].apply(
            lambda x: '' if pd.isna(x) or str(x) == 'nan' else str(x).strip()
        )
        
        # Ensure the DataFrame columns are in the exact order we want (plus source id)
        final_df['source_row_id'] = display_df['source_row_id'] if 'source_row_id' in display_df.columns else list(range(len(final_df)))

        # Sort by date (parsed) then amount as tie-breaker
        _sort_dt = pd.to_datetime(final_df['Transaction Date'], errors='coerce')
        final_df = final_df.assign(_sort_dt=_sort_dt).sort_values(by=['_sort_dt', 'Amount'], ascending=[True, True]).drop(columns=['_sort_dt'])

        column_order = ['Transaction Date', 'Amount', 'Clean Vendor', 'Category', 'Description/Memo', 'source_row_id']
        final_df = final_df[column_order]
        
        # Memo enrichment (optional, bounded cost: uses OPENAI_API_KEY if present)
        enable_memo_enrichment = bool(config_override.get('enable_memo_enrichment', False))
        memo_budget = float(config_override.get('memo_cost_cap_per_request', 0.2))
        use_gpt5 = enable_memo_enrichment and os.getenv('OPENAI_API_KEY')

        # Prepare enrichment batch if enabled
        enrich_indices = []
        enrich_payload = []
        if use_gpt5:
            for i in final_df.index:
                memo_val = str(final_df.at[i, 'Description/Memo'] or '')
                if _memo_needs_enrichment(memo_val):
                    vendor = str(final_df.at[i, 'Clean Vendor'] or '')
                    category = str(final_df.at[i, 'Category'] or '')
                    amount = float(final_df.at[i, 'Amount'] or 0.0)
                    # Add a rules fallback now; LLM can override
                    fallback = _infer_rule_memo(vendor, category)
                    final_df.at[i, 'Description/Memo'] = fallback
                    enrich_indices.append(i)
                    enrich_payload.append({
                        'vendor': vendor,
                        'category': category,
                        'amount': amount,
                        'notes': ''
                    })
            # Rough cap: assume ~$0.01 per memo; trim if over budget
            max_rows = int(max(0, min(len(enrich_payload), memo_budget / 0.01))) if memo_budget > 0 else 0
            enrich_payload = enrich_payload[:max_rows]
            enrich_indices = enrich_indices[:max_rows]
            if enrich_payload:
                enriched = _gpt5_batch_enrich(enrich_payload, {})
                for idx, memo in zip(enrich_indices, enriched):
                    if memo and memo.strip():
                        final_df.at[idx, 'Description/Memo'] = memo.strip()

        # Convert to JSON preserving order by manually building the response
        cleaned_records = []
        for idx, row in final_df.iterrows():
            record = {}
            for col in column_order:
                value = row[col]
                if pd.isna(value):
                    if col == 'Clean Vendor':
                        value = '[Vendor Missing]'
                    elif col == 'Category':
                        value = '[Category Missing]'
                    else:
                        value = None
                record[col] = value
            # Add raw fields expected by tests/clients (use normalized display fields to ensure quality)
            record['standardized_vendor'] = str(final_df.at[idx, 'Clean Vendor'])
            record['category'] = str(final_df.at[idx, 'Category'])
            cleaned_records.append(record)
        
        display_df = final_df
        
        # --- Strict Mode Quality Gates ---
        strict_cfg = summary.get('strict_config') or {}
        # Merge request-level config into strict_cfg
        if isinstance(config_override, dict):
            strict_cfg = {
                'strict_mode': bool(config_override.get('strict_mode', False)),
                'auto_remediate': bool(config_override.get('auto_remediate', True)),
                'reject_on_fail': bool(config_override.get('reject_on_fail', True)),
                'thresholds': dict({
                    'vendor_non_missing_min': 0.98,
                    'category_non_other_min': 0.98,
                    'date_parse_min': 0.99,
                    'amount_valid_min': 1.0,
                    'duplicate_max': 0.01,
                }, **(config_override.get('thresholds') or {})),
                'backfill_caps': dict({
                    'vendor_cost_cap': 0.5,
                    'category_cost_cap': 0.5,
                }, **(config_override.get('backfill_caps') or {})),
            }

        quality_report = _compute_quality_metrics(display_df)

        def _meets(got: float, key: str, is_max: bool = False) -> bool:
            thr = float(strict_cfg['thresholds'].get(key))
            return got <= thr if is_max else got >= thr

        failed = []
        if strict_cfg.get('strict_mode', False):
            if not _meets(quality_report['vendor_non_missing_rate'], 'vendor_non_missing_min'):
                failed.append('vendor_non_missing_min')
            if not _meets(quality_report['category_non_other_rate'], 'category_non_other_min'):
                failed.append('category_non_other_min')
            if not _meets(quality_report['date_parse_rate'], 'date_parse_min'):
                failed.append('date_parse_min')
            if not _meets(quality_report['amount_valid_rate'], 'amount_valid_min'):
                failed.append('amount_valid_min')
            if not _meets(quality_report['duplicate_rate'], 'duplicate_max', is_max=True):
                failed.append('duplicate_max')

            remediation = {'vendor_updates': 0, 'category_updates': 0, 'ai_cost_spent': 0.0}

            # Auto-remediation if allowed and failed
            if failed and strict_cfg.get('auto_remediate', True):
                try:
                    subset_vendor_idx = [i for i, r in enumerate(cleaned_records) if (str(r.get('standardized_vendor','')).strip() in ('', '[Vendor Missing]'))]
                    vendor_budget = float(strict_cfg['backfill_caps'].get('vendor_cost_cap', 0.5))
                    if subset_vendor_idx and vendor_budget > 0:
                        # Build subset payload with original inputs reconstituted from display_df
                        subset_rows = []
                        for i in subset_vendor_idx:
                            subset_rows.append({
                                'Date': str(display_df.at[i, 'Transaction Date'] or ''),
                                'Merchant': str(pre_normalized_vendor.at[i] if i in pre_normalized_vendor.index else ''),
                                'Amount': float(display_df.at[i, 'Amount'] or 0.0),
                                'Notes': str(display_df.at[i, 'Description/Memo'] or ''),
                                'source_row_id': int(display_df.at[i, 'source_row_id'] if 'source_row_id' in display_df.columns else i),
                            })
                        # Call self API for vendor-only pass
                        import requests as _rq
                        resp = _rq.post(
                            f"http://localhost:{APP_CONFIG['port']}/process",
                            json={
                                'data': subset_rows,
                                'config': {
                                    'preserve_schema': False,
                                    'use_real_llm': True,
                                    'enable_ai': True,
                                    'ai_vendor_enabled': True,
                                    'ai_category_enabled': False,
                                    'enable_transaction_intelligence': False,
                                    'thresholds': {},
                                    'backfill_caps': {},
                                    'reject_on_fail': False,
                                }
                            }, timeout=120
                        )
                        if resp.status_code == 200:
                            bf = resp.json().get('cleaned_data', [])
                            sid_to_vendor = {}
                            for r in bf:
                                sid = r.get('source_row_id')
                                vend = str(r.get('standardized_vendor','')).strip()
                                if sid is not None and vend and vend != '[Vendor Missing]':
                                    sid_to_vendor[int(sid)] = vend
                            # Apply updates
                            for idx, rec in enumerate(cleaned_records):
                                sid = rec.get('source_row_id', idx)
                                if sid in sid_to_vendor:
                                    rec['standardized_vendor'] = sid_to_vendor[sid]
                                    # Update display df too
                                    display_df.at[idx, 'Clean Vendor'] = sid_to_vendor[sid]
                                    remediation['vendor_updates'] += 1
                        # Approximate cost per vendor call
                        remediation['ai_cost_spent'] += 0.01 * len(sid_to_vendor)
                    # Recompute metrics after vendor remediation
                    quality_report = _compute_quality_metrics(display_df)
                    # If categories still fail and we allow category remediation
                    if (not _meets(quality_report['category_non_other_rate'], 'category_non_other_min')):
                        # Category remediation is optional; skipping here to keep scope minimal
                        pass
                except Exception as _rem_err:
                    logger.warning(f"Auto-remediation failed: {_rem_err}")

            # Re-evaluate failures
            failed = []
            if not _meets(quality_report['vendor_non_missing_rate'], 'vendor_non_missing_min'):
                failed.append('vendor_non_missing_min')
            if not _meets(quality_report['category_non_other_rate'], 'category_non_other_min'):
                failed.append('category_non_other_min')
            if not _meets(quality_report['date_parse_rate'], 'date_parse_min'):
                failed.append('date_parse_min')
            if not _meets(quality_report['amount_valid_rate'], 'amount_valid_min'):
                failed.append('amount_valid_min')
            if not _meets(quality_report['duplicate_rate'], 'duplicate_max', is_max=True):
                failed.append('duplicate_max')

            if failed and strict_cfg.get('reject_on_fail', True):
                # Provide a concise failure payload
                sample_issues = []
                for i, rec in enumerate(cleaned_records[:10]):
                    reasons = []
                    if str(rec.get('standardized_vendor','')).strip() in ('', '[Vendor Missing]'):
                        reasons.append('vendor_missing')
                    cat = str(rec.get('category','')).strip()
                    if not cat or cat not in ALLOWED_CATEGORIES or cat == 'Other':
                        reasons.append('category_not_specific')
                    dt = str(rec.get('Transaction Date') or '')
                    if pd.isna(pd.to_datetime(dt, errors='coerce')):
                        reasons.append('date_invalid')
                    amt = rec.get('Amount')
                    try:
                        _ = float(amt)
                    except Exception:
                        reasons.append('amount_invalid')
                    if reasons:
                        sample_issues.append({'source_row_id': rec.get('source_row_id', i), 'reasons': reasons})
                return jsonify({
                    'error': 'quality_thresholds_not_met',
                    'quality_report': quality_report,
                    'thresholds': strict_cfg['thresholds'],
                    'failed_thresholds': failed,
                    'sample_problem_rows': sample_issues,
                    'request_id': request_id
                }), 422

        # --- Phase 2: Structure output for HITL (clean_data + review_queue) ---
        hitl_conf_threshold = float(config_override.get('hitl_conf_threshold', 0.95))
        hitl_generic_categories = set(config_override.get('hitl_generic_categories', ['Other', 'General Services']))
        hitl_high_value_amount = float(config_override.get('hitl_high_value_amount', 1000.0))

        def _row_processing_notes(row_idx: int) -> Dict[str, Any]:
            src = ''
            conf = None
            try:
                if 'category_source' in cleaned_df.columns:
                    src = str(cleaned_df.at[row_idx, 'category_source'])
                elif 'vendor_source' in cleaned_df.columns:
                    src = str(cleaned_df.at[row_idx, 'vendor_source'])
            except Exception:
                src = ''
            try:
                if 'category_confidence' in cleaned_df.columns and pd.notna(cleaned_df.at[row_idx, 'category_confidence']):
                    conf = float(cleaned_df.at[row_idx, 'category_confidence'])
                elif 'vendor_confidence' in cleaned_df.columns and pd.notna(cleaned_df.at[row_idx, 'vendor_confidence']):
                    conf = float(cleaned_df.at[row_idx, 'vendor_confidence'])
            except Exception:
                conf = None
            return {'source': src or 'rule', 'confidence': float(conf) if conf is not None else 0.0}

        clean_data = []
        review_queue = []

        for idx in final_df.index:
            try:
                row = final_df.loc[idx]
                notes = _row_processing_notes(idx)
                amount_val = float(row['Amount'] or 0.0)
                category_val = str(row['Category'] or '')
                reasons = []
                if notes.get('confidence', 0.0) < hitl_conf_threshold:
                    reasons.append('LOW_CONFIDENCE')
                if category_val in hitl_generic_categories:
                    reasons.append('GENERIC_CATEGORY')
                if abs(amount_val) > hitl_high_value_amount:
                    reasons.append('HIGH_VALUE')

                base_record = {
                    'source_row_id': int(row['source_row_id']) if 'source_row_id' in row else int(idx),
                    'Transaction Date': row['Transaction Date'],
                    'Amount': amount_val,
                    'Clean Vendor': row['Clean Vendor'],
                    'Category': category_val,
                    'Description/Memo': row['Description/Memo'],
                    'processing_notes': notes
                }

                if reasons:
                    review_queue.append({
                        'source_row_id': base_record['source_row_id'],
                        'original_description': base_record['Description/Memo'],
                        'suggested_vendor': base_record['Clean Vendor'],
                        'suggested_category': base_record['Category'],
                        'amount': base_record['Amount'],
                        'processing_notes': {**notes, 'reason_for_review': reasons[0]}
                    })
                else:
                    clean_data.append(base_record)
            except Exception:
                # On any unexpected issue, be conservative and send to review
                try:
                    review_queue.append({
                        'source_row_id': int(idx),
                        'original_description': str(final_df.at[idx, 'Description/Memo']) if 'Description/Memo' in final_df.columns else '',
                        'suggested_vendor': str(final_df.at[idx, 'Clean Vendor']) if 'Clean Vendor' in final_df.columns else '',
                        'suggested_category': str(final_df.at[idx, 'Category']) if 'Category' in final_df.columns else '',
                        'amount': float(final_df.at[idx, 'Amount']) if 'Amount' in final_df.columns else 0.0,
                        'processing_notes': {'source': 'rule', 'confidence': 0.0, 'reason_for_review': 'SYSTEM_ERROR'}
                    })
                except Exception:
                    pass

        total_rows = len(final_df)
        rows_for_review = len(review_queue)
        rows_cleaned = len(clean_data)
        vendor_cov = quality_report.get('vendor_non_missing_rate', 0.0)
        ai_calls = int(processing_summary.get('llm_calls', 0))
        total_cost = float(cost_analysis.get('total_cost', 0.0))

        hitl_response = {
            'processing_level': 'SUCCESS_FULL_ENRICHMENT',
            'summary_report': {
                'total_rows': total_rows,
                'rows_cleaned': rows_cleaned,
                'rows_for_review': rows_for_review,
                'vendor_coverage': f"{vendor_cov*100:.1f}%",
                'ai_calls': ai_calls,
                'total_cost': f"${total_cost:.3f}"
            },
            'clean_data': clean_data,
            'review_queue': review_queue
        }

        return jsonify(hitl_response)

    except Exception as e:
        logger.error(f"[{request_id}] Unhandled error: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.', 'details': str(e)}), 500

if __name__ == '__main__':
    try:
        get_llm_client() # Initialize client on startup
        logger.info(f"API starting on port {APP_CONFIG['port']}")
        app.run(host='0.0.0.0', port=APP_CONFIG['port'], debug=APP_CONFIG['debug'])
    except Exception as e:
        logger.critical(f"Failed to start API: {e}", exc_info=True)
