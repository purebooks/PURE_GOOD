#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ProcessingSource(Enum):
    RULE_BASED = "rule_based"
    CACHE = "cache"
    LLM = "llm"


@dataclass
class ProcessingResult:
    value: Any
    source: ProcessingSource
    confidence: float
    explanation: str
    processing_time: float
    cost: float = 0.0


class IntelligentCache:
    def __init__(self, max_cache_size: int = 2000):
        self.vendor_cache: Dict[str, Dict[str, Any]] = {}
        self.category_cache: Dict[str, Dict[str, Any]] = {}
        self.max_cache_size = max_cache_size
        self.cache_stats = {
            'vendor_hits': 0, 'vendor_misses': 0,
            'category_hits': 0, 'category_misses': 0,
            'cache_evictions': 0
        }

    def _normalize_vendor_name(self, vendor: str) -> str:
        if not vendor:
            return ""
        v = str(vendor).strip()
        # Strip payment processor prefixes and trailing ids
        import re
        v = re.sub(r'^(PAYPAL\s*\*|SQ\s*\*|TST\s*\*|PP\*)', '', v, flags=re.IGNORECASE).strip()
        v = re.sub(r'(\s*#\d+\s*$)', '', v, flags=re.IGNORECASE).strip()
        v = re.sub(r'(\s*\*STORE\s*\d+\s*$)', '', v, flags=re.IGNORECASE).strip()
        return v.upper()

    def _amount_bucket(self, amount: float) -> str:
        a = abs(float(amount or 0.0))
        if a == 0: return 'zero'
        if a < 5: return 'micro'
        if a < 15: return 'small'
        if a < 50: return 'medium'
        if a < 200: return 'large'
        if a < 1000: return 'xlarge'
        return 'huge'

    def _key(self, payload: Dict[str, Any]) -> str:
        processed: Dict[str, Any] = {}
        for k, v in payload.items():
            if k == 'vendor':
                processed[k] = self._normalize_vendor_name(str(v))
            elif k == 'amount':
                processed['amount_bucket'] = self._amount_bucket(float(v or 0.0))
            else:
                processed[k] = v
        return hashlib.md5(json.dumps(processed, sort_keys=True).encode()).hexdigest()

    def _evict_if_needed(self, cache: Dict[str, Any]):
        if len(cache) >= self.max_cache_size:
            # naive LRU: drop 20% oldest
            to_remove = max(1, len(cache) // 5)
            for k in list(cache.keys())[:to_remove]:
                cache.pop(k, None)
            self.cache_stats['cache_evictions'] += to_remove

    def get_vendor(self, vendor: str) -> Optional[Dict[str, Any]]:
        key = self._key({'vendor': vendor})
        if key in self.vendor_cache:
            self.cache_stats['vendor_hits'] += 1
            return self.vendor_cache[key]
        self.cache_stats['vendor_misses'] += 1
        return None

    def set_vendor(self, vendor: str, result: Dict[str, Any]):
        self._evict_if_needed(self.vendor_cache)
        key = self._key({'vendor': vendor})
        self.vendor_cache[key] = result

    def get_category(self, vendor: str, amount: float, description_hash: str) -> Optional[Dict[str, Any]]:
        key = self._key({'vendor': vendor, 'amount': amount, 'desc': description_hash})
        if key in self.category_cache:
            self.cache_stats['category_hits'] += 1
            return self.category_cache[key]
        self.cache_stats['category_misses'] += 1
        return None

    def set_category(self, vendor: str, amount: float, description_hash: str, result: Dict[str, Any]):
        self._evict_if_needed(self.category_cache)
        key = self._key({'vendor': vendor, 'amount': amount, 'desc': description_hash})
        self.category_cache[key] = result

    def stats(self) -> Dict[str, Any]:
        total_v = self.cache_stats['vendor_hits'] + self.cache_stats['vendor_misses']
        total_c = self.cache_stats['category_hits'] + self.cache_stats['category_misses']
        return {
            'vendor_hit_rate': (self.cache_stats['vendor_hits'] / total_v) if total_v else 0,
            'category_hit_rate': (self.cache_stats['category_hits'] / total_c) if total_c else 0,
            **self.cache_stats,
        }


class LLMTracker:
    def __init__(self):
        self.cost = 0.0
        self.calls = 0
        self.success = 0
        self.fail = 0
        self.time_total = 0.0

    def track(self, elapsed: float, est_cost: float, ok: bool):
        self.calls += 1
        self.time_total += elapsed
        self.cost += est_cost
        if ok: self.success += 1
        else: self.fail += 1

    def summary(self) -> Dict[str, Any]:
        avg = (self.time_total / self.calls) if self.calls else 0.0
        return {'calls': self.calls, 'success': self.success, 'fail': self.fail, 'total_cost': self.cost, 'avg_time': avg}


class LLMAssistant:
    """Post-clean LLM assist/apply pass for vendor/category only (schema-preserving)."""

    def __init__(self, allowed_categories: List[str]):
        self.allowed_categories = set(allowed_categories)
        self.cache = IntelligentCache()
        self.tracker = LLMTracker()

    # ---------- Targeting ----------
    def _looks_like_product(self, text: str) -> bool:
        if not text: return False
        low = text.lower()
        if any(k in low for k in ["mug","candle","paper","ink","stapler","journal","notebook","ticket","gift", "inventory", "wholesale"]):
            return True
        words = text.split()
        return 2 <= len(words) <= 5 and sum(1 for w in words if w[:1].isupper()) >= len(words) - 1

    def _description_hash(self, s: str) -> str:
        return hashlib.md5((s or "").strip().encode()).hexdigest()

    def select_targets(
        self,
        df: pd.DataFrame,
        ambiguous_categories: List[str],
        *,
        vendor_col: Optional[str] = None,
        category_col: Optional[str] = None,
        description_col: Optional[str] = None,
    ) -> List[int]:
        amb = set(x.lower() for x in ambiguous_categories)
        targets: List[int] = []
        for i, row in df.iterrows():
            vendor = str(
                (row.get(vendor_col) if vendor_col else row.get('Vendor/Customer', row.get('merchant', '')))
                or ''
            ).strip()
            category = str(
                (row.get(category_col) if category_col else row.get('Category', row.get('category', '')))
                or ''
            ).strip()
            desc = str(
                (row.get(description_col) if description_col else row.get('Description', row.get('description', '')))
                or ''
            ).strip()
            if not vendor or self._looks_like_product(vendor):
                targets.append(i)
                continue
            if (not category) or (category.lower() in amb):
                if vendor or desc:
                    targets.append(i)
        return targets

    # ---------- OpenAI call ----------
    def _call_openai_batch(self, rows: List[Dict[str, Any]], model: str, temperature: float) -> List[Dict[str, Any]]:
        prompt_items = []
        for r in rows:
            prompt_items.append({
                'idx': r['idx'],
                'date': r['date'],
                'amount': r['amount'],
                'vendor_text': r['vendor_text'],
                'description': r['description'],
                'current_category': r['current_category']
            })
        user_prompt = (
            "You are an accounting assistant. For each transaction, return JSON only with keys: "
            "vendor (string, standardized business name or ''), "
            "category (from the allowed list), "
            "confidence (0..1), "
            "description (optional concise memo if the original description is blank, else '').\n"
            "Allowed categories: "
        )
        # We will pass allowed categories in the messages content
        try:
            from openai import OpenAI  # type: ignore
            import os
            api_key = os.getenv('OPENAI_API_KEY', '')
            if not api_key:
                raise RuntimeError('OPENAI_API_KEY missing')
            client = OpenAI(api_key=api_key)
            content = (
                f"Allowed: {sorted(list(self.allowed_categories))}\n"
                f"Return ONLY a JSON array; no explanations.\n" + json.dumps(prompt_items)
            )
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": "You produce strictly valid JSON only."},
                    {"role": "user", "content": user_prompt + content},
                ],
                max_tokens=800,
            )
            elapsed = time.time() - t0
            text = (resp.choices[0].message.content or '').strip()
            start = text.find('['); end = text.rfind(']') + 1
            arr = json.loads(text[start:end] if start != -1 and end > start else text)
            self.tracker.track(elapsed, 0.0025 * len(rows), True)
            return arr if isinstance(arr, list) else []
        except Exception as e:
            logger.warning(f"OpenAI call failed; falling back to no-op: {e}")
            self.tracker.track(0.0, 0.0, False)
            # Fallback heuristic: no suggestions
            return []

    # ---------- Application ----------
    def enhance(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], int, Dict[str, Any]]:
        mode = str(config.get('ai_mode', 'assist')).lower()
        threshold = float(config.get('ai_confidence_threshold', 0.9))
        ambiguous_categories = list(config.get('ai_ambiguous_categories', ["Other", "Miscellaneous", "Sales"]))
        batch_size = int(config.get('ai_batch_size', 10))
        cost_cap = float(config.get('ai_cost_cap_per_request', 0.10))
        est_per_row = float(config.get('ai_estimated_cost_per_row', 0.0025))
        model = str(config.get('ai_model', 'gpt-5-mini'))
        temperature = float(config.get('ai_temperature', 0.2))
        allow_apply_in_preserve = bool(config.get('ai_preserve_schema_apply', False))

        suggestions: Dict[int, Dict[str, Any]] = {}
        applied = 0

        # Resolve roles dynamically
        try:
            from field_resolver import FieldResolver
            roles = FieldResolver(df).roles()
        except Exception:
            roles = type('R', (), {'vendor': None, 'category': None, 'description': None})()

        targets = self.select_targets(
            df,
            ambiguous_categories,
            vendor_col=roles.vendor,
            category_col=roles.category,
            description_col=roles.description,
        )
        if not targets:
            return suggestions, applied, self._stats()

        # Prepare batches with caching
        # Deterministic pre-rules (apply mode only): map 'cloud' to Software & Technology when category is blank/ambiguous
        if mode == 'apply':
            amb_set = set(x.lower() for x in ambiguous_categories)
            kept_targets: List[int] = []
            for idx in targets:
                try:
                    cat = str(df.at[idx, roles.category]) if (getattr(roles, 'category', None) and roles.category in df.columns) else (
                        str(df.at[idx, 'Category']) if 'Category' in df.columns else ''
                    )
                except Exception:
                    cat = ''
                is_amb = (not str(cat).strip()) or str(cat).strip().lower() in amb_set
                vendor_text = str(
                    df.at[idx, roles.vendor] if (getattr(roles, 'vendor', None) and roles.vendor in df.columns) else (
                        df.at[idx, 'Vendor/Customer'] if 'Vendor/Customer' in df.columns else (
                            df.at[idx, 'merchant'] if 'merchant' in df.columns else ''
                        )
                    )
                )
                desc_text = str(
                    df.at[idx, roles.description] if (getattr(roles, 'description', None) and roles.description in df.columns) else (
                        df.at[idx, 'Description'] if 'Description' in df.columns else (
                            df.at[idx, 'description'] if 'description' in df.columns else ''
                        )
                    )
                )
                low_blob = f"{vendor_text} {desc_text}".lower()
                if is_amb and 'cloud' in low_blob:
                    target_cat_col = roles.category if (getattr(roles, 'category', None) and roles.category in df.columns) else ('Category' if 'Category' in df.columns else None)
                    if target_cat_col:
                        df.at[idx, target_cat_col] = 'Software & Technology'
                        applied += 1
                    # Do not send to LLM anymore
                else:
                    kept_targets.append(idx)
            targets = kept_targets

        # After pre-rules, continue building rows for LLM
        rows_for_llm: List[Dict[str, Any]] = []
        for idx in targets:
            r = df.loc[idx]
            vendor = str(
                (r.get(roles.vendor) if getattr(roles, 'vendor', None) else r.get('Vendor/Customer', r.get('merchant', '')))
                or ''
            ).strip()
            amount = abs(float(r.get('Amount', r.get('amount', 0)) or 0))
            desc = str(
                (r.get(roles.description) if getattr(roles, 'description', None) else r.get('Description', r.get('description', '')))
                or ''
            ).strip()
            cat = str(
                (r.get(roles.category) if getattr(roles, 'category', None) else r.get('Category', r.get('category', '')))
                or ''
            ).strip()
            desc_hash = self._description_hash(desc)
            cached = self.cache.get_category(vendor, amount, desc_hash)
            if cached:
                suggestions[idx] = {'vendor': cached.get('vendor', ''), 'category': cached.get('category', ''), 'confidence': cached.get('confidence', 0.0), 'source': 'cache'}
                continue
            rows_for_llm.append({
                'idx': idx,
                'date': str(r.get('Date', r.get('date', '')) or ''),
                'amount': amount,
                'vendor_text': vendor,
                'description': desc,
                'current_category': cat,
                'desc_hash': desc_hash,
            })

        # Cost gate
        projected = est_per_row * len(rows_for_llm)
        if projected > cost_cap:
            # Trim rows by priority: ambiguous first already; enforce by amount desc
            rows_for_llm.sort(key=lambda x: x['amount'], reverse=True)
            max_rows = int(max(0, cost_cap // est_per_row))
            rows_for_llm = rows_for_llm[:max_rows]

        # Batch call
        for i in range(0, len(rows_for_llm), batch_size):
            batch = rows_for_llm[i:i + batch_size]
            if not batch:
                break
            results = self._call_openai_batch(batch, model=model, temperature=temperature)
            # Map results back
            for j, res in enumerate(results):
                try:
                    idx = batch[j]['idx']
                    vendor_s = str(res.get('vendor', '') or '').strip()
                    category_s = str(res.get('category', '') or '').strip()
                    conf = float(res.get('confidence', 0.0) or 0.0)
                    desc_s = str(res.get('description', '') or '').strip()
                    if category_s not in self.allowed_categories:
                        category_s = ''
                    suggestions[idx] = {
                        'vendor': vendor_s,
                        'category': category_s,
                        'description': desc_s,
                        'confidence': conf,
                        'source': 'llm'
                    }
                    # Cache
                    self.cache.set_category(
                        batch[j]['vendor_text'], batch[j]['amount'], batch[j]['desc_hash'],
                        {'vendor': vendor_s, 'category': category_s, 'confidence': conf}
                    )
                except Exception:
                    continue

        # Apply if configured (schema-preserving)
        if mode == 'apply' and (allow_apply_in_preserve or True):  # external guard should enforce preserve policy
            for idx, sug in suggestions.items():
                if sug.get('confidence', 0.0) < threshold:
                    continue
                # Apply only to blank/ambiguous targets
                target_cat_col = None
                if getattr(roles, 'category', None) and roles.category in df.columns:
                    target_cat_col = roles.category
                elif 'Category' in df.columns:
                    target_cat_col = 'Category'
                categ = str(df.at[idx, target_cat_col]) if target_cat_col else ''
                is_amb = (not categ.strip()) or categ.strip().lower() in set(x.lower() for x in ambiguous_categories)
                if target_cat_col and sug.get('category') and is_amb:
                    df.at[idx, target_cat_col] = sug['category']
                    applied += 1
                # Vendor fill only if blank or product-like
                target_vendor_col = None
                if getattr(roles, 'vendor', None) and roles.vendor in df.columns:
                    target_vendor_col = roles.vendor
                elif 'Vendor/Customer' in df.columns:
                    target_vendor_col = 'Vendor/Customer'
                elif 'Merchant' in df.columns:
                    target_vendor_col = 'Merchant'
                if target_vendor_col:
                    vend = str(df.at[idx, target_vendor_col] or '').strip()
                    if (not vend) or self._looks_like_product(vend):
                        if sug.get('vendor'):
                            df.at[idx, target_vendor_col] = sug['vendor']
                            applied += 1
                # Description fill only if blank
                target_desc_col = None
                if getattr(roles, 'description', None) and roles.description in df.columns:
                    target_desc_col = roles.description
                elif 'Description' in df.columns:
                    target_desc_col = 'Description'
                if target_desc_col:
                    cur_desc = str(df.at[idx, target_desc_col] or '').strip()
                    if not cur_desc and sug.get('description'):
                        df.at[idx, target_desc_col] = sug['description'][:64]
                        applied += 1

        return suggestions, applied, self._stats()

    def _stats(self) -> Dict[str, Any]:
        return {
            'cache': self.cache.stats(),
            'tracker': self.tracker.summary(),
        }


