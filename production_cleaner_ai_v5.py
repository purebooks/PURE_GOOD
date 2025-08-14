import pandas as pd
import numpy as np
import re
import json
import logging
import hashlib
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
try:
    from llm_client_v2 import LLMClient  # Use enhanced version if available
except ImportError:
    from llm_client import LLMClient  # Fallback to original

# Import Configuration Manager
try:
    from config_manager import get_config_for_intent, ConfigManager
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    print("Warning: Configuration manager not available. Using legacy config.")
    CONFIG_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)

class ProcessingSource(Enum):
    """Sources of processing decisions"""
    RULE_BASED = "rule_based"
    CACHE = "cache"
    LLM = "llm"

@dataclass
class ProcessingResult:
    """Result of any processing operation with source tracking"""
    value: Any
    source: ProcessingSource
    confidence: float
    explanation: str
    processing_time: float
    cost: float = 0.0

class IntelligentCache:
    """Advanced caching system with keyed storage and statistics"""
    
    def __init__(self):
        self.vendor_cache = {}
        self.category_cache = {}
        self.intelligence_cache = {}
        self.cache_stats = {
            'vendor_hits': 0,
            'vendor_misses': 0,
            'category_hits': 0,
            'category_misses': 0,
            'intelligence_hits': 0,
            'intelligence_misses': 0
        }
    
    def _generate_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate consistent cache keys"""
        sorted_data = json.dumps(data, sort_keys=True)
        return hashlib.md5(sorted_data.encode()).hexdigest()
    
    def get_vendor_cache(self, vendor: str) -> Optional[Dict[str, Any]]:
        """Get vendor standardization from cache"""
        key = self._generate_cache_key({'vendor': vendor})
        if key in self.vendor_cache:
            self.cache_stats['vendor_hits'] += 1
            return self.vendor_cache[key]
        self.cache_stats['vendor_misses'] += 1
        return None
    
    def set_vendor_cache(self, vendor: str, result: Dict[str, Any]):
        """Cache vendor standardization result"""
        key = self._generate_cache_key({'vendor': vendor})
        self.vendor_cache[key] = result
    
    def get_category_cache(self, vendor: str, amount: float) -> Optional[Dict[str, Any]]:
        """Get category classification from cache"""
        key = self._generate_cache_key({'vendor': vendor, 'amount': amount})
        if key in self.category_cache:
            self.cache_stats['category_hits'] += 1
            return self.category_cache[key]
        self.cache_stats['category_misses'] += 1
        return None
    
    def set_category_cache(self, vendor: str, amount: float, result: Dict[str, Any]):
        """Cache category classification result"""
        key = self._generate_cache_key({'vendor': vendor, 'amount': amount})
        self.category_cache[key] = result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_vendor = self.cache_stats['vendor_hits'] + self.cache_stats['vendor_misses']
        total_category = self.cache_stats['category_hits'] + self.cache_stats['category_misses']
        
        return {
            'vendor_hit_rate': self.cache_stats['vendor_hits'] / total_vendor if total_vendor > 0 else 0,
            'category_hit_rate': self.cache_stats['category_hits'] / total_category if total_category > 0 else 0,
            'total_vendor_requests': total_vendor,
            'total_category_requests': total_category,
            'cache_size': len(self.vendor_cache) + len(self.category_cache)
        }

class LLMTracker:
    """Comprehensive LLM cost, time, and performance tracking"""
    
    def __init__(self):
        self.cost_tracker = {
            'total_cost': 0.0,
            'vendor_standardization_cost': 0.0,
            'category_classification_cost': 0.0,
            'transaction_intelligence_cost': 0.0
        }
        self.time_tracker = {
            'total_time': 0.0,
            'vendor_standardization_time': 0.0,
            'category_classification_time': 0.0,
            'transaction_intelligence_time': 0.0
        }
        self.performance_tracker = {
            'total_llm_calls': 0,
            'successful_llm_calls': 0,
            'failed_llm_calls': 0,
            'average_response_time': 0.0
        }
    
    def track_llm_call(self, operation: str, cost: float, time: float, success: bool = True):
        """Track LLM call metrics"""
        self.cost_tracker[f'{operation}_cost'] += cost
        self.cost_tracker['total_cost'] += cost
        self.time_tracker[f'{operation}_time'] += time
        self.time_tracker['total_time'] += time
        self.performance_tracker['total_llm_calls'] += 1
        
        if success:
            self.performance_tracker['successful_llm_calls'] += 1
        else:
            self.performance_tracker['failed_llm_calls'] += 1
        
        # Update average response time
        total_calls = self.performance_tracker['successful_llm_calls'] + self.performance_tracker['failed_llm_calls']
        if total_calls > 0:
            self.performance_tracker['average_response_time'] = self.time_tracker['total_time'] / total_calls
    
    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get comprehensive tracking summary"""
        return {
            'cost_analysis': self.cost_tracker,
            'performance_metrics': self.time_tracker,
            'llm_performance': self.performance_tracker,
            'cost_per_call': self.cost_tracker['total_cost'] / self.performance_tracker['total_llm_calls'] if self.performance_tracker['total_llm_calls'] > 0 else 0
        }

class AIEnhancedProductionCleanerV5:
    """
    AI-Enhanced Production Financial Cleaner v5.0
    Advanced LLM Flow with Intelligent Processing
    Features:
    - Advanced LLM flow with source tracking
    - Intelligent caching system
    - Comprehensive cost and performance tracking
    - Transaction intelligence (separate from CSV)
    - Enhanced DataFrame with source attribution
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None, config: Optional[Dict[str, Any]] = None, 
                 llm_client: Optional[LLMClient] = None, user_intent: Optional[str] = None):
        self.cleaner_version = "v5.0"
        self.df = (df.copy() if df is not None else pd.DataFrame())
        self.user_intent = user_intent
        
        # Validate and set configuration (now supports intent-based configs)
        self.config = self._validate_and_set_config(config or {}, user_intent)
        
        # Initialize AI client
        self.llm_client = llm_client or LLMClient(use_mock=False)
        
        # Initialize advanced components
        self.cache = IntelligentCache()
        self.tracker = LLMTracker()
        self.processing_history = []
        self._claude_enhancement_applied = False
        self.preserve_schema = bool(self.config.get('preserve_schema', False))
        # Per-request LLM budget tracking
        self.llm_cost_cap = float(self.config.get('llm_cost_cap_per_request', 0.05))
        self.llm_cost_spent = 0.0
        self.llm_estimated_cost_per_row = float(self.config.get('llm_estimated_cost_per_row', 0.005))
        
        # Initialize advanced LLM processor with our enhanced vendor rules
        try:
            from advanced_llm_components import AdvancedLLMProcessor
            self.advanced_processor = AdvancedLLMProcessor(self.llm_client, self.config)
        except ImportError:
            logger.warning("Advanced LLM components not available, using basic processing")
            self.advanced_processor = None

        # If explicitly forcing LLM for testing, bypass the advanced processor
        if self.config.get('force_llm_for_testing', False):
            logger.info("force_llm_for_testing enabled: bypassing AdvancedLLMProcessor to route to LLM paths")
            self.advanced_processor = None
        
        # Processing statistics
        self.stats = {
            'ai_requests': 0,
            'ai_cost': 0.0,
            'cache_hits': 0,
            'processing_time': 0.0,
            'rows_processed': len(self.df),
            'vendor_standardizations': 0,
            'category_classifications': 0,
            'transaction_intelligence': 0
        }
        
        logger.info(f"AI-Enhanced Production Cleaner {self.cleaner_version} initialized with {len(self.df)} rows")

    def _clean_vendor_for_processing(self, vendor_text: Any) -> str:
        """Strip processor prefixes/suffixes and corporate suffixes for better matching."""
        try:
            s = '' if vendor_text is None else str(vendor_text)
        except Exception:
            s = ''
        if not s:
            return ''
        prev = None
        while prev != s:
            prev = s
            # leading processors and AUTO PAY
            s = re.sub(r'^(AUTO\s*PAY\s+)', '', s, flags=re.IGNORECASE).strip()
            s = re.sub(r'^(PAYPAL\s*\*|SQ\s*\*|TST\s*\*)', '', s, flags=re.IGNORECASE).strip()
            # trailing artifacts
            s = re.sub(r'(\s*\*STORE\s*\d+\s*$)', '', s, flags=re.IGNORECASE).strip()
            s = re.sub(r'(\s*#\d+\s*$)', '', s, flags=re.IGNORECASE).strip()
            s = re.sub(r'(\s*\.COM\s*$)', '', s, flags=re.IGNORECASE).strip()
            s = re.sub(r'(\s*ONLINE\s*$)', '', s, flags=re.IGNORECASE).strip()
            s = re.sub(r'\b(inc|llc|corp|co|ltd)\.?\s*$', '', s, flags=re.IGNORECASE).strip()
        return ' '.join(s.split())

    def _validate_and_set_config(self, user_config: Dict[str, Any], user_intent: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced configuration system with intent-based templates"""
        
        # Legacy defaults for backward compatibility
        legacy_defaults = {
            'duplicate_threshold': 0.85,
            'outlier_z_threshold': 3.0,
            'chunk_size': 10000,
            'enable_ai': True,
            'ai_confidence_threshold': 0.85,
            'use_ai_for_unmatched_only': False,
            'ai_vendor_enabled': True,
            'ai_category_enabled': True,
            'ai_analysis_enabled': False,
            'enable_transaction_intelligence': False,
            'enable_parallel_processing': True,
            'max_workers': 8,
            'enable_source_tracking': True,
            # Hybrid LLM routing defaults
            'llm_trigger_amount': 0.0,
            'llm_parallel_conf_threshold': 0.9,
            'llm_route_unknown': True,
            'llm_max_enhancements': 180,
            'llm_batch_size': 10,
            'llm_cost_cap_per_request': 1.5,
            'llm_estimated_cost_per_row': 0.01,
            # Targeting controls
            'aggressive_vendor_backfill': True,
            'llm_always_reprocess_categories': ['Other', 'General Merchandise', 'General Services'],
            # Guarantee some LLM coverage when desired
            'llm_minimum_fraction': 0.0,
            'llm_minimum_count': 0,
            # Accuracy-first mode and SLOs
            'accuracy_first': True,
            'quality_slo_proper_min': 0.97,   # tuned for optimized cost/quality
            'quality_slo_none_max': 0.0,      # no None allowed
            'emergency_llm_cap': 0.0          # no emergency spend in optimized mode
        }
        
        # Try to load intent-based configuration
        if CONFIG_MANAGER_AVAILABLE and user_intent:
            try:
                logger.info(f"Loading intent-based configuration for: {user_intent}")
                intent_config = get_config_for_intent(user_intent, user_config)
                
                # Convert intent-based config to legacy format for compatibility
                converted_config = self._convert_intent_config_to_legacy(intent_config, legacy_defaults)
                
                # Store the full intent config for advanced features
                self.intent_config = intent_config
                
                # Ensure explicit user flags are preserved (e.g., force_llm_for_testing) and LLM routing knobs
                if 'force_llm_for_testing' in user_config:
                    converted_config['force_llm_for_testing'] = bool(user_config['force_llm_for_testing'])
                if 'enable_parallel_processing' in user_config:
                    converted_config['enable_parallel_processing'] = bool(user_config['enable_parallel_processing'])
                # Preserve other runtime controls from user_config
                passthrough_keys = [
                    'max_workers',
                    'enable_transaction_intelligence',
                    'llm_trigger_amount',
                    'llm_parallel_conf_threshold',
                    'llm_route_unknown',
                    'llm_max_enhancements',
                    'llm_batch_size',
                    'llm_cost_cap_per_request',
                    'llm_estimated_cost_per_row'
                ]
                for k in passthrough_keys:
                    if k in user_config:
                        converted_config[k] = user_config[k]

                logger.info(f"Successfully loaded intent configuration: {intent_config.get('intent', 'unknown')}")
                return converted_config
                
            except Exception as e:
                logger.warning(f"Failed to load intent configuration: {e}, falling back to legacy")
        
        # Fallback to legacy configuration
        logger.info("Using legacy configuration system")
        self.intent_config = None
        
        # Merge user config with defaults
        config = {**legacy_defaults, **user_config}
        
        # Type validation
        for key in ['duplicate_threshold', 'outlier_z_threshold', 'ai_confidence_threshold']:
            if key in config:
                config[key] = float(config[key])
        if 'chunk_size' in config:
            config['chunk_size'] = int(config['chunk_size'])
        for key in ['enable_ai', 'use_ai_for_unmatched_only', 'ai_vendor_enabled', 
                   'ai_category_enabled', 'ai_analysis_enabled', 'enable_transaction_intelligence', 
                   'enable_source_tracking']:
            if key in config:
                config[key] = bool(config[key])
        
        return config
    
    def _convert_intent_config_to_legacy(self, intent_config: Dict[str, Any], legacy_defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Convert intent-based configuration to legacy format for compatibility"""
        
        # Start with legacy defaults
        config = legacy_defaults.copy()
        
        # Map AI processing settings
        ai_settings = intent_config.get('ai_processing', {})
        config['ai_confidence_threshold'] = ai_settings.get('confidence_threshold', 0.7)
        config['enable_ai'] = True  # Always enable AI for intent-based processing
        config['ai_vendor_enabled'] = True
        config['ai_category_enabled'] = True

        # Map Claude enhancement knobs when present in templates
        claude_settings = ai_settings.get('claude_enhancement', {}) if isinstance(ai_settings, dict) else {}
        if isinstance(claude_settings, dict):
            if 'reprocess_confidence_threshold' in claude_settings:
                config['llm_parallel_conf_threshold'] = float(claude_settings['reprocess_confidence_threshold'])
            if 'always_reprocess_categories' in claude_settings:
                config['llm_always_reprocess_categories'] = list(claude_settings['always_reprocess_categories'])
            if 'enable_aggressive_vendor_backfill' in claude_settings:
                config['aggressive_vendor_backfill'] = bool(claude_settings['enable_aggressive_vendor_backfill'])
            if 'llm_cost_cap_per_request' in claude_settings:
                config['llm_cost_cap_per_request'] = float(claude_settings['llm_cost_cap_per_request'])
        
        # Map analysis features
        analysis_features = intent_config.get('analysis_features', {})
        config['ai_analysis_enabled'] = analysis_features.get('calculate_category_totals', False)
        config['enable_transaction_intelligence'] = analysis_features.get('detect_trends', True)
        
        # Map data quality settings
        data_quality = intent_config.get('data_quality', {})
        if 'duplicate_threshold' in data_quality:
            config['duplicate_threshold'] = data_quality['duplicate_threshold']
        if 'outlier_threshold' in data_quality:
            config['outlier_z_threshold'] = data_quality['outlier_threshold']
        
        # Enable parallel processing if specified
        if ai_settings.get('enable_parallel_processing', True):
            config['chunk_size'] = 5000  # Smaller chunks for parallel processing
        
        # Force LLM testing mode if requested
        if ai_settings.get('force_llm_for_testing', False):
            config['use_ai_for_unmatched_only'] = False
        
        logger.debug(f"Converted intent config to legacy format: {list(config.keys())}")
        return config
    
    def _apply_intent_specific_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply intent-specific processing rules to the data"""
        if not self.intent_config:
            return df
        
        # Apply column filtering based on intent
        cleaning_focus = self.intent_config.get('cleaning_focus', {})
        
        # Prioritize certain columns
        columns_to_prioritize = cleaning_focus.get('columns_to_prioritize', [])
        if columns_to_prioritize:
            # Ensure prioritized columns are processed first
            available_priority_cols = [col for col in columns_to_prioritize if col in df.columns]
            other_cols = [col for col in df.columns if col not in available_priority_cols]
            df = df[available_priority_cols + other_cols]
            logger.info(f"Prioritized columns based on intent: {available_priority_cols}")
        
        # Filter out unwanted columns
        columns_to_ignore = cleaning_focus.get('columns_to_ignore', [])
        if columns_to_ignore:
            cols_to_drop = [col for col in columns_to_ignore if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                logger.info(f"Removed columns based on intent: {cols_to_drop}")
        
        # Apply filtering if specified
        filtering = self.intent_config.get('filtering', {})
        target_category = filtering.get('target_category')
        if target_category:
            logger.info(f"Intent requests filtering for category: {target_category}")
            # This will be applied during categorization phase
        
        return df
    
    def _get_intent_category_rules(self) -> Dict[str, str]:
        """Get custom category rules from intent configuration"""
        if not self.intent_config:
            return {}
        
        categorization = self.intent_config.get('categorization', {})
        custom_rules = categorization.get('custom_category_rules', {})
        
        logger.debug(f"Loaded {len(custom_rules)} custom category rules from intent")
        return custom_rules
    
    def _get_intent_vendor_emphasis(self) -> List[str]:
        """Get vendor processing emphasis from intent configuration"""
        if not self.intent_config:
            return []
        
        vendor_processing = self.intent_config.get('vendor_processing', {})
        emphasis = vendor_processing.get('vendor_emphasis', [])
        
        logger.debug(f"Intent vendor emphasis: {emphasis}")
        return emphasis
    
    def _should_apply_intent_filtering(self, category: str) -> bool:
        """Check if a transaction matches intent-based filtering criteria"""
        if not self.intent_config:
            return True
        
        filtering = self.intent_config.get('filtering', {})
        target_category = filtering.get('target_category')
        
        if target_category:
            # For "find X spending" intents, only include matching categories
            return target_category.lower() in category.lower()
        
        return True
    
    def _apply_intent_result_formatting(self, output: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Apply intent-specific formatting to the results"""
        if not self.intent_config:
            return output
        
        output_prefs = self.intent_config.get('output_preferences', {})
        
        # Add intent-specific summary
        intent_summary = {
            'intent': self.intent_config.get('intent', 'unknown'),
            'description': self.intent_config.get('description', ''),
            'processing_mode': output_prefs.get('format', 'standard')
        }
        
        # Apply category-specific insights for budget analysis
        if self.intent_config.get('intent') == 'budget_analysis':
            intent_summary.update(self._generate_budget_insights(df))
        
        # Apply category filtering insights for find_category_spending  
        elif self.intent_config.get('intent') == 'find_category_spending':
            intent_summary.update(self._generate_category_spending_insights(df))
        
        # Apply subscription insights for subscription_audit
        elif self.intent_config.get('intent') == 'subscription_audit':
            intent_summary.update(self._generate_subscription_insights(df))
        
        # Add intent summary to output
        output['intent_summary'] = intent_summary
        
        logger.info(f"Applied intent-specific formatting for: {intent_summary['intent']}")
        return output
    
    def _generate_budget_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate budget analysis specific insights"""
        insights = {}
        
        if 'category' in df.columns and 'amount' in df.columns:
            # Calculate spending by category
            category_spending = df.groupby('category')['amount'].agg(['sum', 'count', 'mean']).round(2)
            insights['category_breakdown'] = category_spending.to_dict('index')
            
            # Find highest spending categories
            top_categories = category_spending['sum'].abs().nlargest(5)
            insights['top_spending_categories'] = top_categories.to_dict()
            
            # Calculate total spending
            insights['total_spending'] = float(df['amount'].sum())
            insights['average_transaction'] = float(df['amount'].mean())
            insights['transaction_count'] = len(df)
        
        return insights
    
    def _generate_category_spending_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate category-specific spending insights"""
        insights = {}
        
        filtering = self.intent_config.get('filtering', {})
        target_category = filtering.get('target_category')
        
        if target_category and 'category' in df.columns:
            # Filter for target category
            category_df = df[df['category'].str.contains(target_category, case=False, na=False)]
            
            if len(category_df) > 0:
                insights['target_category'] = target_category
                insights['matching_transactions'] = len(category_df)
                insights['total_spent'] = float(category_df['amount'].sum())
                insights['average_amount'] = float(category_df['amount'].mean())
                
                if 'merchant' in category_df.columns:
                    vendor_breakdown = category_df.groupby('merchant')['amount'].sum().abs().nlargest(5)
                    insights['top_vendors'] = vendor_breakdown.to_dict()
        
        return insights
    
    def _generate_subscription_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate subscription audit specific insights"""
        insights = {}
        
        if 'merchant' in df.columns and 'amount' in df.columns:
            # Look for recurring patterns (simplified)
            recurring_vendors = df['merchant'].value_counts()
            potential_subscriptions = recurring_vendors[recurring_vendors >= 2]  # Appears 2+ times
            
            insights['potential_subscriptions'] = len(potential_subscriptions)
            insights['subscription_candidates'] = potential_subscriptions.head(10).to_dict()
            
            # Calculate potential monthly cost
            subscription_df = df[df['merchant'].isin(potential_subscriptions.index)]
            if len(subscription_df) > 0:
                insights['estimated_monthly_cost'] = float(subscription_df['amount'].abs().sum())
        
        return insights

    @contextmanager
    def timer(self, operation_name: str):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            logger.info(f"{operation_name} completed in {elapsed:.2f} seconds")

    def process_data(self, data: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None,
                     runtime_config: Optional[Dict[str, Any]] = None) -> Union[Tuple[pd.DataFrame, Dict[str, Any]], Dict[str, Any]]:
        """Main processing pipeline with advanced LLM flow and intent-based processing.
        Compatibility: when called with data/config arguments, returns a dict with
        'cleaned_data' (DataFrame) and 'summary_report' including 'llm_tracker'.
        Without arguments, returns (enhanced_df, output_dict) for backward compatibility.
        """
        # Optional runtime overrides
        if data is not None:
            try:
                self.df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()
            except Exception:
                self.df = pd.DataFrame()
        if runtime_config:
            try:
                # Merge runtime config into existing config (shallow)
                self.config.update(runtime_config)
            except Exception:
                pass
        with self.timer("Total advanced LLM processing"):
            # Step 0: Apply intent-specific data filtering and preparation
            processed_df = self._apply_intent_specific_processing(self.df)
            
            # Step 1: Input → Schema Validation → Column Analysis
            schema_analysis = self._validate_schema_and_analyze(processed_df)
            logger.info(f"Schema analysis: {schema_analysis}")
            
            # Step 2: Row Loop with Intelligent Processing (now with intent awareness)
            enhanced_df, processing_results = self._process_rows_intelligently(processed_df)
            
            # Step 3: Generate Comprehensive Output (enhanced with intent context)
            output = self._generate_advanced_output(enhanced_df, processing_results, schema_analysis)
            
            # Step 4: Apply intent-specific result formatting
            output = self._apply_intent_result_formatting(output, enhanced_df)
            
            # Step 5: POST-PROCESSING CLAUDE ENHANCEMENT (Phase 1)
            enhanced_df = self._post_process_with_claude(enhanced_df)
            
            # If called with arguments, return a compatibility dict expected by some tests
            if data is not None or runtime_config is not None:
                # Ensure 'llm_tracker' presence in summary_report
                summary = dict(output.get('summary_report', {}))
                if 'llm_tracker' not in summary:
                    # Compose minimal tracker from available metrics
                    total_cost = 0.0
                    try:
                        # Prefer combined cost if present
                        if isinstance(summary.get('cost_analysis'), dict):
                            total_cost = float(summary['cost_analysis'].get('total_cost', 0.0))
                    except Exception:
                        total_cost = 0.0
                    summary['llm_tracker'] = {
                        'total_calls': int(self.tracker.performance_tracker.get('total_llm_calls', 0)),
                        'total_cost': float(total_cost)
                    }
                    output['summary_report'] = summary
                return {
                    'cleaned_data': enhanced_df,
                    'summary_report': output['summary_report']
                }
            # Default tuple return
            return enhanced_df, output

    def _post_process_with_claude(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 1: Post-processing with Claude for Other categories and low confidence cases.
        This strategic enhancement targets the remaining edge cases for 99%+ accuracy.
        """
        # Respect non-destructive mode: do not add/alter columns beyond in-place cleaning
        if getattr(self, 'preserve_schema', False):
            logger.info("Skipping Claude enhancement in preserve_schema mode")
            return df
        if getattr(self, '_claude_enhancement_applied', False):
            logger.info("Skipping Claude enhancement (already applied)")
            return df
        if self.llm_client.use_mock:
            logger.info("Skipping Claude post-processing in mock mode")
            return df
            
        # Respect per-request LLM budget
        if self.llm_cost_cap <= 0:
            logger.info("Skipping Claude enhancement due to zero LLM budget cap")
            return df

        logger.info("Starting Claude post-processing enhancement...")
        start_time = time.time()
        
        # STEP 1: Identify rows that need Claude enhancement
        needs_enhancement = self._identify_enhancement_targets(df)
        # Prioritize rows already labeled Other/None, then low-confidence, then unknown/processor vendors, then high amount
        def priority_index(idx: int) -> int:
            row = df.iloc[idx]
            cat = str(row.get('category', '') or '').strip().lower()
            cat_conf = float(row.get('category_confidence', 0.0) or 0.0)
            vendor = str(row.get('standardized_vendor', '') or '').strip()
            is_proc = bool(re.match(r'^(PAYPAL\s*\*|SQ\s*\*|TST\s*\*)', vendor, flags=re.IGNORECASE))
            p = 0
            if (not cat) or cat == 'none' or cat == 'other':
                p -= 100
            if cat_conf < float(self.config.get('llm_parallel_conf_threshold', 0.9)):
                p -= 50
            if is_proc:
                p -= 25
            if abs(float(row.get('amount', 0) or 0)) >= float(self.config.get('llm_trigger_amount', 0.0)):
                p -= 10
            return p
        needs_enhancement = sorted(needs_enhancement, key=priority_index)
        
        if len(needs_enhancement) == 0:
            logger.info("No rows need Claude enhancement - all classifications are high confidence")
            return df
        
        logger.info(f"Found {len(needs_enhancement)} rows for Claude enhancement")
        
        # STEP 2: Batch process with Claude for efficiency (cap to avoid long timeouts)
        max_items = int(self.config.get('llm_max_enhancements', 25))
        # Ensure a minimum coverage if requested (fraction or count)
        min_frac = float(self.config.get('llm_minimum_fraction', 0.0))
        min_count = int(self.config.get('llm_minimum_count', 0))
        guarantee = max(int(len(df) * min_frac), min_count)
        if guarantee > 0 and len(needs_enhancement) < guarantee:
            # Add additional indices to meet minimum, prioritizing high amounts
            candidates = list(set(range(len(df))) - set(needs_enhancement))
            candidates.sort(key=lambda i: abs(float(df.at[i, 'amount'] if 'amount' in df.columns else df.at[i, 'Amount'] if 'Amount' in df.columns else 0)), reverse=True)
            needs_enhancement += candidates[:max(0, guarantee - len(needs_enhancement))]
        target_indices = needs_enhancement[:max_items]
        enhanced_rows = self._enhance_with_claude_batch(target_indices, df)
        
        # STEP 3: Update dataframe with enhanced results
        df = self._apply_claude_enhancements(df, enhanced_rows)
        
        processing_time = time.time() - start_time
        logger.info(f"Claude post-processing completed in {processing_time:.2f}s, enhanced {len(enhanced_rows)} rows")
        
        self._claude_enhancement_applied = True

        # STEP 4: Accuracy-first SLO check and emergency expansion if needed
        if bool(self.config.get('accuracy_first', False)):
            try:
                proper = 0
                other = 0
                none = 0
                for _, row in df.iterrows():
                    cat = str(row.get('category', '') or '').strip()
                    if not cat or cat == 'None':
                        none += 1
                    elif cat == 'Other':
                        other += 1
                    else:
                        proper += 1
                total = max(1, len(df))
                proper_rate = proper / total
                none_rate = none / total
                slo_proper = float(self.config.get('quality_slo_proper_min', 0.98))
                slo_none = float(self.config.get('quality_slo_none_max', 0.01))

                if (proper_rate < slo_proper or none_rate > slo_none):
                    extra = float(self.config.get('emergency_llm_cap', 0.30))
                    if extra > 0:
                        logger.info(f"Accuracy-first SLO not met (proper={proper_rate:.3f}, none={none_rate:.3f}). Applying emergency extra budget ${extra:.2f}.")
                        self.llm_cost_cap += extra
                        # Recompute remaining candidates (focus on Other/None/low-confidence)
                        needs_enhancement = []
                        conf_threshold = float(self.config.get('llm_parallel_conf_threshold', 0.9))
                        for idx, row in df.iterrows():
                            cat = str(row.get('category', '') or '').strip()
                            cat_conf = float(row.get('category_confidence', 0.0) or 0.0)
                            if (not cat or cat == 'None' or cat == 'Other' or cat_conf < conf_threshold):
                                needs_enhancement.append(idx)
                        # Run another batched pass under new budget
                        extra_rows = self._enhance_with_claude_batch(needs_enhancement, df)
                        df = self._apply_claude_enhancements(df, extra_rows)
            except Exception as e:
                logger.warning(f"Accuracy-first SLO evaluation failed: {e}")

        return df

    def _remaining_llm_budget(self) -> float:
        return max(0.0, self.llm_cost_cap - self.llm_cost_spent)

    def _can_spend_llm(self, rows: int = 1) -> bool:
        estimated = self.llm_estimated_cost_per_row * max(1, rows)
        return self._remaining_llm_budget() >= estimated

    @contextmanager
    def _temporary_mock_mode(self, enable_mock: bool):
        prev = getattr(self.llm_client, 'use_mock', False)
        try:
            setattr(self.llm_client, 'use_mock', enable_mock)
            yield
        finally:
            setattr(self.llm_client, 'use_mock', prev)

    def _identify_enhancement_targets(self, df: pd.DataFrame) -> List[int]:
        """
        Hybrid targeting for post-processing with Claude (real LLM):
        - High-value transactions above llm_trigger_amount
        - Low-confidence classifications below llm_parallel_conf_threshold
        - Category is empty/None/Other
        - Optional: route unknown vendors via vendor confidence threshold when available
        """
        targets = []
        trigger_amount = float(self.config.get('llm_trigger_amount', 300.0))
        conf_threshold = float(self.config.get('llm_parallel_conf_threshold', 0.9))
        route_unknown = bool(self.config.get('llm_route_unknown', True))
        always_cats = [str(c).strip().lower() for c in self.config.get('llm_always_reprocess_categories', ['Other'])]
        aggressive_backfill = bool(self.config.get('aggressive_vendor_backfill', False))

        for idx, row in df.iterrows():
            raw_cat = row.get('category', '')
            category = str(raw_cat) if raw_cat is not None else ''
            category = category.strip()
            cat_conf = float(row.get('category_confidence', 1.0) or 1.0)
            amount = abs(float(row.get('amount', 0) or 0)) if pd.notna(row.get('amount')) else 0.0
            raw_vendor = row.get('standardized_vendor', '')
            vendor = str(raw_vendor) if raw_vendor is not None else ''
            vendor = vendor.strip()
            ven_conf = float(row.get('vendor_confidence', 1.0) or 1.0)

            # 1) High-value
            if amount >= trigger_amount:
                targets.append(idx)
                continue

            # 2) Always-reprocess categories regardless of confidence
            if category.lower() in always_cats:
                targets.append(idx)
                continue

            # 3) Low-confidence classification
            if cat_conf < conf_threshold:
                targets.append(idx)
                continue

            # 4) Missing/Other category (fallback general check)
            if category in ('', 'None', 'Other'):
                targets.append(idx)
                continue

            # 5) Unknown/processor vendor routing (use vendor confidence and aggressive backfill)
            is_processor = bool(re.match(r'^(PAYPAL\s*\*|SQ\s*\*|TST\s*\*)', vendor, flags=re.IGNORECASE))
            is_unknown_vendor = vendor.strip() == '' or vendor.strip().lower() in {'unknown vendor', 'unknown', 'other'}
            if route_unknown and (ven_conf < conf_threshold or (aggressive_backfill and (is_unknown_vendor or is_processor))):
                targets.append(idx)
                continue

        logger.info(f"Hybrid Claude targeting: {len(targets)} out of {len(df)} transactions selected")
        return targets

    def _enhance_with_claude_batch(self, target_indices: List[int], df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """Use Claude to enhance target rows (no explanations, cost-capped, batched)"""
        enhanced_rows = {}
        
        # Process in batches per config
        batch_size = int(self.config.get('llm_batch_size', 5))
        cost_cap = float(self.config.get('llm_cost_cap_per_request', 0.05))
        est_cost_per_row = float(self.config.get('llm_estimated_cost_per_row', 0.005))
        spent_estimate = self.llm_cost_spent
        for i in range(0, len(target_indices), batch_size):
            batch_indices = target_indices[i:i + batch_size]
            # Respect cost cap (rough estimate)
            projected = spent_estimate + est_cost_per_row * len(batch_indices)
            if projected > cost_cap:
                logger.info(f"Stopping Claude batches due to cost cap: projected ${projected:.3f} > ${cost_cap:.2f}")
                break
            batch_rows = []
            
            # Prepare batch for Claude
            for idx in batch_indices:
                row = df.iloc[idx]
                batch_rows.append({
                    'index': idx,
                    'merchant': row.get('standardized_vendor', row.get('Merchant', '')),
                    'current_category': row.get('category', 'Other'),
                    'amount': abs(float(row.get('amount', 0))) if pd.notna(row.get('amount')) else 0,
                    'confidence': row.get('category_confidence', 0.5)
                })
            
            # Make Claude call for this batch
            # Make real call only if we have budget
            if not self._can_spend_llm(len(batch_rows)):
                logger.info("Skipping batch due to insufficient LLM budget")
                break
            batch_results = self._call_claude_for_batch(batch_rows)
            
            # Store results
            for result in batch_results:
                enhanced_rows[result['index']] = result
            spent_estimate += est_cost_per_row * len(batch_results)
            self.llm_cost_spent = spent_estimate
        
        return enhanced_rows

    def _call_claude_for_batch(self, batch_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make a single Claude call to enhance a batch of rows"""
        try:
            call_start = time.time()
            # Build the prompt for Claude
            prompt = self._build_claude_enhancement_prompt(batch_rows)
            
            # Make the LLM call
            response_text = self.llm_client._make_llm_call(prompt)
            llm_time = time.time() - call_start
            
            # Parse the response
            enhanced_results = self._parse_claude_response(response_text, batch_rows)

            # Track estimated cost and calls
            est_cost_per_row = float(self.config.get('llm_estimated_cost_per_row', 0.005))
            est_cost = est_cost_per_row * len(batch_rows)
            self.tracker.track_llm_call('category_classification', est_cost, llm_time, True)
            # Count rows touched by LLM in ai_requests; cost added to stats for quick insights
            self.stats['ai_requests'] += len(batch_rows)
            self.stats['ai_cost'] += est_cost
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Claude batch enhancement failed: {e}")
            # Return original data if Claude fails
            return [{
                'index': row['index'],
                'enhanced_category': row['current_category'],
                'explanation': f"Claude enhancement failed, keeping original: {row['current_category']}",
                'confidence': row['confidence'],
                'enhanced': False
            } for row in batch_rows]

    def _build_claude_enhancement_prompt(self, batch_rows: List[Dict[str, Any]]) -> str:
        """Build a concise prompt for Claude (no explanations requested)"""
        
        # Create the merchant list
        merchants_list = []
        for i, row in enumerate(batch_rows):
            merchants_list.append(
                f"{i+1}. {row['merchant']} (current: {row['current_category']}, amount: ${row['amount']:.2f}, confidence: {row['confidence']:.2f})"
            )
        
        prompt = f"""You are a business expense classifier.

MERCHANTS:
{chr(10).join(merchants_list)}

Categories: Software & Technology, Meals & Entertainment, Travel & Transportation, Office Supplies & Equipment, Professional Services, Banking & Finance, Utilities & Rent, Marketing & Advertising, Employee Benefits, Insurance & Legal, Other.

Return ONLY a JSON array of objects without explanations:
[
  {{"merchant":"exact_merchant_name","category":"best_category","confidence":0.95}},
  ...
]

JSON:"""
        
        return prompt

    def _parse_claude_response(self, response_text: str, batch_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse Claude's JSON response and map back to our row indices"""
        try:
            # Extract JSON from response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_content = response_text[start_idx:end_idx]
                claude_results = json.loads(json_content)
            else:
                claude_results = json.loads(response_text)
            
            # Map results back to our indices (no explanations expected)
            enhanced_results = []
            for i, row in enumerate(batch_rows):
                if i < len(claude_results):
                    claude_result = claude_results[i]
                    enhanced_results.append({
                        'index': row['index'],
                        'enhanced_category': claude_result.get('category', row['current_category']),
                        'explanation': 'AI-enhanced classification',
                        'confidence': claude_result.get('confidence', 0.9),
                        'enhanced': True
                    })
                else:
                    # Fallback if Claude didn't return enough results
                    enhanced_results.append({
                        'index': row['index'],
                        'enhanced_category': row['current_category'],
                        'explanation': 'No enhancement available',
                        'confidence': row['confidence'],
                        'enhanced': False
                    })
            
            return enhanced_results
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Claude JSON response: {e}")
            # Return fallback results
            return [{
                'index': row['index'],
                'enhanced_category': row['current_category'],
                'explanation': 'JSON parse failed, keeping original',
                'confidence': row['confidence'],
                'enhanced': False
            } for row in batch_rows]

    def _apply_claude_enhancements(self, df: pd.DataFrame, enhanced_rows: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
        """Apply Claude enhancements back to the dataframe"""
        enhanced_count = 0
        
        for idx, enhancement in enhanced_rows.items():
            if enhancement['enhanced']:
                # Update the category
                df.at[idx, 'category'] = enhancement['enhanced_category']
                df.at[idx, 'category_confidence'] = enhancement['confidence']
                df.at[idx, 'category_source'] = 'claude_enhanced'
                
                # Drop explanations entirely in new mode (keep compatibility column empty if present)
                if 'category_explanation' in df.columns:
                    df.at[idx, 'category_explanation'] = ''
                
                enhanced_count += 1
                
                # Safely retrieve original merchant for logging (columns may vary)
                try:
                    original_merchant = df.at[idx, 'standardized_vendor'] if 'standardized_vendor' in df.columns else None
                except Exception:
                    original_merchant = None
                if not original_merchant:
                    try:
                        original_merchant = df.at[idx, 'Merchant'] if 'Merchant' in df.columns else ''
                    except Exception:
                        original_merchant = ''
                logger.info(f"Claude enhanced: {original_merchant} → {enhancement['enhanced_category']} | {enhancement['explanation']}")
        
        logger.info(f"Successfully enhanced {enhanced_count} rows with Claude")
        return df

    def _validate_schema_and_analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data schema and perform comprehensive column analysis"""
        logger.info("Performing schema validation and column analysis")
        
        # Schema validation
        schema_validation = {
            'valid': True,
            'issues': []
        }
        
        # Check for required columns
        if len(df.columns) == 0:
            schema_validation['valid'] = False
            schema_validation['issues'].append("No columns found")
        
        # Column analysis
        column_analysis = {
            'vendor_columns': self._find_vendor_columns(df),
            'amount_columns': self._find_amount_columns(df),
            'category_columns': self._find_category_columns(df),
            'date_columns': self._find_date_columns(df),
            'description_columns': self._find_description_columns(df)
        }
        
        # Assess processing capabilities
        processing_capabilities = self._assess_processing_capabilities(column_analysis)
        
        return {
            'schema_valid': schema_validation['valid'],
            'schema_issues': schema_validation['issues'],
            'column_mapping': column_analysis,
            'processing_capabilities': processing_capabilities
        }

    def _process_rows_intelligently(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Process rows with intelligent flow: Rule > Cache > LLM"""
        logger.info("Starting intelligent row processing")
        
        # Determine if we should use parallel processing
        # Honor config flag directly (parallelism is beneficial even for ~100 rows with batching)
        use_parallel = self.config.get('enable_parallel_processing', True)
        max_workers = self.config.get('max_workers', min(4, len(df) // 5))
        
        if use_parallel:
            logger.info(f"Using parallel processing with {max_workers} workers for {len(df)} rows")
            enhanced_df, processing_results = self._process_rows_parallel(df, max_workers)
            # After parallel pass, optionally enhance with Claude per hybrid rules
            enhanced_df = self._post_process_with_claude(enhanced_df)
            return enhanced_df, processing_results
        else:
            logger.info("Using sequential processing")
            return self._process_rows_sequential(df)
    
    def _process_rows_sequential(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Sequential processing (original method)"""
        enhanced_df = df.copy()
        processing_results = []
        
        for idx, row in df.iterrows():
            row_result = {
                'row_index': idx,
                'vendor_standardization': None,
                'category_classification': None,
                'transaction_intelligence': None
            }
            
            # Find relevant columns
            vendor_col = self._find_vendor_column(df)
            amount_col = self._find_amount_column(df)
            
            # Process vendor standardization
            if vendor_col and self.config['ai_vendor_enabled']:
                vendor_input = row.get(vendor_col)
                # --- ROBUSTNESS FIX for NaN/missing values ---
                if pd.isna(vendor_input):
                    vendor = ''
                else:
                    vendor = str(vendor_input)
                
                if vendor and vendor.strip():
                    vendor_result = self._process_vendor_standardization(vendor, row)
                    if self.preserve_schema:
                        # Write standardized value back to the original vendor column only
                        enhanced_df.at[idx, vendor_col] = vendor_result.value
                    else:
                        enhanced_df.at[idx, 'standardized_vendor'] = vendor_result.value
                    if self.config['enable_source_tracking']:
                        enhanced_df.at[idx, 'vendor_source'] = vendor_result.source.value
                        enhanced_df.at[idx, 'vendor_confidence'] = vendor_result.confidence
                    row_result['vendor_standardization'] = {
                        'value': vendor_result.value,
                        'source': vendor_result.source.value,
                        'confidence': vendor_result.confidence,
                        'explanation': vendor_result.explanation,
                        'processing_time': vendor_result.processing_time,
                        'cost': vendor_result.cost
                    }
                    self.stats['vendor_standardizations'] += 1
            
            # Process category classification
            if vendor_col and amount_col and self.config['ai_category_enabled']:
                # Use standardized vendor name for better category matching
                standardized_vendor = enhanced_df.at[idx, 'standardized_vendor'] if 'standardized_vendor' in enhanced_df.columns else str(enhanced_df.get(vendor_col, row.get(vendor_col, '')))
                # ROBUST AMOUNT PARSING (fixes None category bug)
                amount_val = row.get(amount_col, 0)
                if amount_val == "" or amount_val is None or pd.isna(amount_val):
                    amount = 0.0
                else:
                    try:
                        # Clean common amount formats
                        amount_str = str(amount_val).strip()
                        
                        # Handle parentheses (accounting negative format)
                        if amount_str.startswith('(') and amount_str.endswith(')'):
                            amount_str = '-' + amount_str[1:-1]
                        
                        # Remove common currency symbols and commas
                        amount_str = amount_str.replace('$', '').replace(',', '').replace(' ', '')
                        
                        # Convert to float
                        amount = float(amount_str) if amount_str else 0.0
                    except (ValueError, TypeError):
                        logger.warning(f"Failed to parse amount: {amount_val}, using 0.0")
                        amount = 0.0
                
                # --- ROBUSTNESS FIX for NaN/missing standardized_vendor ---
                if pd.isna(standardized_vendor):
                    sanitized_vendor = ''
                else:
                    sanitized_vendor = str(standardized_vendor)

                if sanitized_vendor:  # REMOVE amount > 0 restriction - classify all vendors
                    category_result = self._process_category_classification(sanitized_vendor, abs(amount), row)
                    # Only set category if a category-like column exists when preserving schema
                    if self.preserve_schema:
                        existing_cat_col = next((c for c in enhanced_df.columns if str(c).lower() in ['category','type','classification','group']), None)
                        if existing_cat_col:
                            enhanced_df.at[idx, existing_cat_col] = category_result.value
                    else:
                        enhanced_df.at[idx, 'category'] = category_result.value
                    if self.config['enable_source_tracking']:
                        if not self.preserve_schema:
                            enhanced_df.at[idx, 'category_source'] = category_result.source.value
                            enhanced_df.at[idx, 'category_confidence'] = category_result.confidence
                    row_result['category_classification'] = {
                        'value': category_result.value,
                        'source': category_result.source.value,
                        'confidence': category_result.confidence,
                        'explanation': category_result.explanation,
                        'processing_time': category_result.processing_time,
                        'cost': category_result.cost
                    }
                    self.stats['category_classifications'] += 1
            
            # Process transaction intelligence (separate from CSV)
            if self.config['enable_transaction_intelligence'] and not self.preserve_schema:
                intelligence = self._process_transaction_intelligence(row)
                row_result['transaction_intelligence'] = intelligence
                self.stats['transaction_intelligence'] += 1
            
            processing_results.append(row_result)
            
            # Log progress for large datasets
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} rows")
        
        return enhanced_df, processing_results

    def _process_rows_parallel(self, df: pd.DataFrame, max_workers: int) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Process rows in parallel using ThreadPoolExecutor"""
        enhanced_df = df.copy()
        processing_results = []
        
        # Split dataframe into chunks for parallel processing
        chunk_size = max(1, len(df) // max_workers)
        chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]
        
        logger.info(f"Processing {len(chunks)} chunks of ~{chunk_size} rows each")
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunk processing tasks
            future_to_chunk = {
                executor.submit(self._process_chunk_worker, chunk, chunk_idx): (chunk, chunk_idx)
                for chunk_idx, chunk in enumerate(chunks)
            }
            
            # Collect results as they complete
            chunk_results = []
            for future in as_completed(future_to_chunk):
                chunk, chunk_idx = future_to_chunk[future]
                try:
                    chunk_result = future.result()
                    chunk_results.append((chunk_idx, chunk_result))
                    logger.info(f"Completed chunk {chunk_idx + 1}/{len(chunks)}")
                except Exception as exc:
                    logger.error(f"Chunk {chunk_idx} generated an exception: {exc}")
                    # Create a fallback result for failed chunk
                    fallback_result = {
                        'enhanced_chunk': chunk,
                        'processing_results': []
                    }
                    chunk_results.append((chunk_idx, fallback_result))
        
        # Sort results by chunk index to maintain order
        chunk_results.sort(key=lambda x: x[0])
        
        # Merge results from all chunks
        all_processing_results = []
        for chunk_idx, result in chunk_results:
            chunk_enhanced = result['enhanced_chunk']
            chunk_processing = result['processing_results']

            # Ensure new columns from chunk are present in the master df
            for col in chunk_enhanced.columns:
                if col not in enhanced_df.columns:
                    try:
                        enhanced_df[col] = pd.NA
                    except Exception:
                        enhanced_df[col] = None

            # Bulk-assign chunk columns back to the master df (preserves new cols like 'standardized_vendor', 'category')
            try:
                enhanced_df.loc[chunk_enhanced.index, chunk_enhanced.columns] = chunk_enhanced.values
            except Exception:
                # Fallback to cell-wise assignment
                for idx in chunk_enhanced.index:
                    for col in chunk_enhanced.columns:
                        enhanced_df.at[idx, col] = chunk_enhanced.at[idx, col]

            # Add chunk processing results
            all_processing_results.extend(chunk_processing)
        
        logger.info(f"Parallel processing complete: {len(all_processing_results)} rows processed")
        return enhanced_df, all_processing_results
    
    def _process_chunk_worker(self, chunk: pd.DataFrame, chunk_idx: int) -> Dict[str, Any]:
        """Worker function to process a chunk of rows with batch LLM calls"""
        logger.info(f"Processing chunk {chunk_idx} with {len(chunk)} rows")
        
        enhanced_chunk = chunk.copy()
        chunk_processing_results = []
        
        # Check if enhanced LLM client supports batch processing
        has_batch_methods = hasattr(self.llm_client, 'process_transaction_batch')
        
        if has_batch_methods and len(chunk) > 5:
            return self._process_chunk_batch(chunk, chunk_idx)
        else:
            return self._process_chunk_sequential(chunk, chunk_idx)
    
    def _process_chunk_batch(self, chunk: pd.DataFrame, chunk_idx: int) -> Dict[str, Any]:
        """Process chunk using batch LLM calls for maximum efficiency"""
        logger.info(f"Batch processing chunk {chunk_idx} with {len(chunk)} rows")
        
        enhanced_chunk = chunk.copy()
        chunk_processing_results = []
        
        # Find relevant columns
        vendor_col = self._find_vendor_column(chunk)
        amount_col = self._find_amount_column(chunk)
        
        if vendor_col and self.config.get('ai_vendor_enabled', False):
            # Prepare batch data for LLM client
            batch_rows = []
            for idx, row in chunk.iterrows():
                batch_rows.append({
                    'merchant': str(row.get(vendor_col, '')),
                    'amount': float(row.get(amount_col, 0)),
                    'description': str(row.get('description', '')),
                    'memo': str(row.get('memo', '')),
                    'row_index': idx
                })
            
            try:
                # Use enhanced LLM client's batch processing
                batch_results = self.llm_client.process_transaction_batch(batch_rows)
                
                # Apply batch results to chunk
                for i, (idx, row) in enumerate(chunk.iterrows()):
                    if i < len(batch_results):
                        result = batch_results[i]
                        
                        # Update enhanced chunk
                        enhanced_chunk.at[idx, 'standardized_vendor'] = result['standardized_vendor']
                        enhanced_chunk.at[idx, 'category'] = result['category']
                        
                        if self.config['enable_source_tracking']:
                            enhanced_chunk.at[idx, 'vendor_source'] = result['processing_source']
                            enhanced_chunk.at[idx, 'vendor_confidence'] = result['confidence']
                            enhanced_chunk.at[idx, 'category_source'] = result['processing_source']
                            enhanced_chunk.at[idx, 'category_confidence'] = result['confidence']
                        
                        # Track processing result
                        row_result = {
                            'row_index': idx,
                            'vendor_standardization': {
                                'value': result['standardized_vendor'],
                                'source': 'llm_batch',
                                'confidence': result['confidence'],
                                'explanation': f"Batch LLM processing: {result['original_merchant']} -> {result['standardized_vendor']}",
                                'processing_time': 0.1,  # Estimated batch time per row
                                'cost': 0.005  # Estimated batch cost per row
                            },
                            'category_classification': {
                                'value': result['category'],
                                'source': 'llm_batch',
                                'confidence': result['confidence'],
                                'explanation': f"Batch category classification: {result['category']}",
                                'processing_time': 0.1,
                                'cost': 0.005
                            },
                            'transaction_intelligence': None
                        }
                        
                        chunk_processing_results.append(row_result)
                        self.stats['vendor_standardizations'] += 1
                        self.stats['category_classifications'] += 1
                
                logger.info(f"Batch processing chunk {chunk_idx} completed successfully")
                
            except Exception as e:
                logger.error(f"Batch processing failed for chunk {chunk_idx}: {e}")
                # Fallback to sequential processing
                return self._process_chunk_sequential(chunk, chunk_idx)
        
        return {
            'enhanced_chunk': enhanced_chunk,
            'processing_results': chunk_processing_results
        }
    
    def _process_chunk_sequential(self, chunk: pd.DataFrame, chunk_idx: int) -> Dict[str, Any]:
        """Sequential processing for chunks (fallback method)"""
        logger.info(f"Sequential processing chunk {chunk_idx} with {len(chunk)} rows")
        
        enhanced_chunk = chunk.copy()
        chunk_processing_results = []
        
        for idx, row in chunk.iterrows():
            row_result = {
                'row_index': idx,
                'vendor_standardization': None,
                'category_classification': None,
                'transaction_intelligence': None
            }
            
            # Find relevant columns
            vendor_col = self._find_vendor_column(chunk)
            amount_col = self._find_amount_column(chunk)
            
            # Process vendor standardization
            if vendor_col and self.config['ai_vendor_enabled']:
                vendor = str(row.get(vendor_col, ''))
                if vendor and vendor.strip():
                    vendor_result = self._process_vendor_standardization(vendor, row)
                    enhanced_chunk.at[idx, 'standardized_vendor'] = vendor_result.value
                    if self.config['enable_source_tracking']:
                        enhanced_chunk.at[idx, 'vendor_source'] = vendor_result.source.value
                        enhanced_chunk.at[idx, 'vendor_confidence'] = vendor_result.confidence
                    row_result['vendor_standardization'] = {
                        'value': vendor_result.value,
                        'source': vendor_result.source.value,
                        'confidence': vendor_result.confidence,
                        'explanation': vendor_result.explanation,
                        'processing_time': vendor_result.processing_time,
                        'cost': vendor_result.cost
                    }
                    self.stats['vendor_standardizations'] += 1
            
            # Process category classification (always classify when vendor text is available; no amount>0 gate)
            if vendor_col and amount_col and self.config['ai_category_enabled']:
                # Use standardized vendor name if available; otherwise original vendor text
                standardized_vendor = enhanced_chunk.at[idx, 'standardized_vendor'] if 'standardized_vendor' in enhanced_chunk.columns else None
                vendor_text = standardized_vendor if (standardized_vendor and str(standardized_vendor).strip()) else str(row.get(vendor_col, ''))
                # Handle empty or invalid amount values
                amount_val = row.get(amount_col, 0)
                if amount_val == "" or amount_val is None:
                    amount = 0.0
                else:
                    try:
                        amount = float(amount_val)
                    except (ValueError, TypeError):
                        amount = 0.0
                if vendor_text:
                    category_result = self._process_category_classification(vendor_text, abs(amount), row)
                    enhanced_chunk.at[idx, 'category'] = category_result.value
                    if self.config['enable_source_tracking']:
                        enhanced_chunk.at[idx, 'category_source'] = category_result.source.value
                        enhanced_chunk.at[idx, 'category_confidence'] = category_result.confidence
                    row_result['category_classification'] = {
                        'value': category_result.value,
                        'source': category_result.source.value,
                        'confidence': category_result.confidence,
                        'explanation': category_result.explanation,
                        'processing_time': category_result.processing_time,
                        'cost': category_result.cost
                    }
                    self.stats['category_classifications'] += 1
            
            # Process transaction intelligence (separate from CSV)
            if self.config['enable_transaction_intelligence']:
                try:
                    intelligence_result = self._process_transaction_intelligence(row)
                    row_result['transaction_intelligence'] = {
                        'value': intelligence_result.value,
                        'source': intelligence_result.source.value,
                        'confidence': intelligence_result.confidence,
                        'explanation': intelligence_result.explanation,
                        'processing_time': intelligence_result.processing_time,
                        'cost': intelligence_result.cost
                    }
                    self.stats['intelligence_generations'] += 1
                except Exception as e:
                    logger.warning(f"Transaction intelligence failed for row {idx}: {e}")
            
            chunk_processing_results.append(row_result)
        
        return {
            'enhanced_chunk': enhanced_chunk,
            'processing_results': chunk_processing_results
        }

    def _process_vendor_standardization(self, vendor: str, row: pd.Series) -> ProcessingResult:
        start_time = time.time()
        generic_vendors = {"client payment", "abc", "unknown", "[vendor missing]", "other", ""}
        description = row.get("Description/Memo", "") if isinstance(row, dict) or hasattr(row, 'get') else ""
        amount = row.get("Amount", 0) if isinstance(row, dict) or hasattr(row, 'get') else 0

        # Step 1: Use advanced processor if available
        if self.advanced_processor:
            return self.advanced_processor.process_vendor_standardization(vendor, row)

        # Step 2: Rule-based (pre-clean for robust matching)
        cleaned_vendor = self._clean_vendor_for_processing(vendor)
        rule_result = self._apply_vendor_rules(cleaned_vendor or vendor)
        if rule_result['matched']:
            # Check for ambiguous/generic/low-confidence
            if (
                rule_result['standardized'].strip().lower() in generic_vendors or
                rule_result['confidence'] < 0.7
            ):
                suggestions = self.llm_client.suggest_vendors_from_description(description, amount)
                main_vendor = suggestions[0] if suggestions else rule_result['standardized']
                return ProcessingResult(
                    value=main_vendor,
                    source=ProcessingSource.LLM,
                    confidence=0.8,
                    explanation=f"AI vendor suggestions: {suggestions}",
                    processing_time=time.time() - start_time
                )
            else:
                return ProcessingResult(
                    value=rule_result['standardized'],
                    source=ProcessingSource.RULE_BASED,
                    confidence=rule_result['confidence'],
                    explanation=rule_result['explanation'],
                    processing_time=time.time() - start_time
                )

        # Step 3: Cache (skip cache path if forcing LLM for testing)
        if self.config.get('force_llm_for_testing', False):
            cache_result = None
        else:
            cache_result = self.cache.get_vendor_cache(cleaned_vendor or vendor)
        if cache_result:
            if (
                cache_result['standardized'].strip().lower() in generic_vendors or
                cache_result['confidence'] < 0.7
            ):
                suggestions = self.llm_client.suggest_vendors_from_description(description, amount)
                main_vendor = suggestions[0] if suggestions else cache_result['standardized']
                return ProcessingResult(
                    value=main_vendor,
                    source=ProcessingSource.LLM,
                    confidence=0.8,
                    explanation=f"AI vendor suggestions: {suggestions}",
                    processing_time=time.time() - start_time
                )
            else:
                return ProcessingResult(
                    value=cache_result['standardized'],
                    source=ProcessingSource.CACHE,
                    confidence=cache_result['confidence'],
                    explanation=cache_result['explanation'],
                    processing_time=time.time() - start_time
                )

        # Step 4: LLM fallback for missing/generic vendor
        if (cleaned_vendor or vendor).strip().lower() in generic_vendors:
            suggestions = self.llm_client.suggest_vendors_from_description(description, amount)
            main_vendor = suggestions[0] if suggestions else vendor
            return ProcessingResult(
                value=main_vendor,
                source=ProcessingSource.LLM,
                confidence=0.8,
                explanation=f"AI vendor suggestions: {suggestions}",
                processing_time=time.time() - start_time
            )

        # Step 5: LLM normal fallback (always LLM if forcing for test)
        try:
            llm_start_time = time.time()
            batch_result = self.llm_client.process_transaction_batch([
                {'merchant': cleaned_vendor or vendor, 'amount': amount, 'description': description}
            ])
            llm_result = batch_result[0]['standardized_vendor']
            llm_time = time.time() - llm_start_time
            self.cache.set_vendor_cache(vendor, {
                'standardized': llm_result,
                'confidence': 0.8,
                'explanation': f"AI-standardized vendor name"
            })
            self.tracker.track_llm_call('vendor_standardization', 0.01, llm_time, True)
            self.stats['ai_requests'] += 1
            self.stats['ai_cost'] += 0.01
            return ProcessingResult(
                value=llm_result,
                source=ProcessingSource.LLM,
                confidence=0.8,
                explanation="AI-powered vendor standardization",
                processing_time=time.time() - start_time,
                cost=0.01
            )
        except Exception as e:
            logger.warning(f"LLM vendor standardization failed for {vendor}: {e}")
            self.tracker.track_llm_call('vendor_standardization', 0.0, time.time() - start_time, False)
            return ProcessingResult(
                value=vendor,
                source=ProcessingSource.RULE_BASED,
                confidence=0.5,
                explanation=f"LLM failed, using original vendor name",
                processing_time=time.time() - start_time
            )

    def _process_category_classification(self, vendor: str, amount: float, row: pd.Series) -> ProcessingResult:
        """Intelligent category classification with source tracking"""
        start_time = time.time()
        
        # Step 1: Use advanced processor if available (has our improved category rules)
        if self.advanced_processor:
            return self.advanced_processor.process_category_classification(vendor, amount, row)
        
        # Fallback to basic rule-based classification
        cleaned_vendor = self._clean_vendor_for_processing(vendor)
        rule_result = self._apply_category_rules(cleaned_vendor or vendor, amount)
        if rule_result['matched']:
            return ProcessingResult(
                value=rule_result['category'],
                source=ProcessingSource.RULE_BASED,
                confidence=rule_result['confidence'],
                explanation=rule_result['explanation'],
                processing_time=time.time() - start_time
            )
        
        # Step 2: Cache lookup
        cache_result = self.cache.get_category_cache(cleaned_vendor or vendor, amount)
        if cache_result:
            return ProcessingResult(
                value=cache_result['category'],
                source=ProcessingSource.CACHE,
                confidence=cache_result['confidence'],
                explanation=cache_result['explanation'],
                processing_time=time.time() - start_time
            )
        
        # Step 3: LLM processing (force LLM path if testing)
        try:
            llm_start_time = time.time()
            # --- CORRECTED FUNCTION CALL ---
            # Build rich description context from available fields
            desc_fields = []
            for key in ['Description', 'description', 'Notes', 'memo', 'Memo', 'Description/Memo']:
                try:
                    val = row.get(key, '') if isinstance(row, dict) or hasattr(row, 'get') else ''
                except Exception:
                    val = ''
                if val:
                    desc_fields.append(str(val))
            description_text = ' | '.join([d for d in desc_fields if d])[:500]
            batch_result = self.llm_client.process_transaction_batch([
                {'merchant': cleaned_vendor or vendor, 'amount': amount, 'description': description_text}
            ])
            llm_result = {
                'category': batch_result[0]['category'],
                'confidence': batch_result[0]['confidence']
            }
            llm_time = time.time() - llm_start_time
            
            # Cache the result
            self.cache.set_category_cache(cleaned_vendor or vendor, amount, {
                'category': llm_result['category'],
                'confidence': llm_result['confidence'],
                'explanation': f"AI-classified category"
            })
            
            # Track LLM call
            self.tracker.track_llm_call('category_classification', 0.01, llm_time, True)
            self.stats['ai_requests'] += 1
            self.stats['ai_cost'] += 0.01
            
            return ProcessingResult(
                value=llm_result['category'],
                source=ProcessingSource.LLM,
                confidence=llm_result['confidence'],
                explanation="AI-powered category classification",
                processing_time=time.time() - start_time,
                cost=0.01
            )
            
        except Exception as e:
            logger.warning(f"LLM category classification failed for {vendor}: {e}")
            self.tracker.track_llm_call('category_classification', 0.0, time.time() - start_time, False)
            
            return ProcessingResult(
                value="Uncategorized",
                source=ProcessingSource.RULE_BASED,
                confidence=0.3,
                explanation=f"LLM failed, using default category",
                processing_time=time.time() - start_time
            )

    def _process_transaction_intelligence(self, row: pd.Series) -> Dict[str, Any]:
        """Generate transaction intelligence (separate from CSV)"""
        intelligence = {
            'tags': self._generate_transaction_tags(row),
            'insights': self._generate_transaction_insights(row),
            'explainability': self._generate_transaction_explanation(row),
            'risk_score': self._calculate_transaction_risk(row),
            'anomaly_detection': self._detect_transaction_anomalies(row)
        }
        
        return intelligence

    def _generate_advanced_output(self, df: pd.DataFrame, processing_results: List[Dict[str, Any]], 
                                schema_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive output with enhanced data, summary, and audit logs"""
        
        # Summary Report
        # Combine metrics from basic tracker and advanced processor (if present)
        adv_tracker = getattr(self.advanced_processor, 'tracker', None)

        # LLM calls
        basic_llm_calls = self.tracker.performance_tracker.get('total_llm_calls', 0) if hasattr(self, 'tracker') else 0
        adv_llm_calls = adv_tracker.performance_tracker.get('total_llm_calls', 0) if adv_tracker else 0
        combined_llm_calls = self.stats['ai_requests'] + basic_llm_calls + adv_llm_calls

        # Cost analysis (sum per key where available)
        combined_cost = dict(self.tracker.cost_tracker)
        if adv_tracker:
            for key, value in adv_tracker.cost_tracker.items():
                combined_cost[key] = combined_cost.get(key, 0.0) + value

        # Performance metrics (sum times)
        combined_perf = dict(self.tracker.time_tracker)
        if adv_tracker:
            for key, value in adv_tracker.time_tracker.items():
                combined_perf[key] = combined_perf.get(key, 0.0) + value

        summary_report = {
            'processing_summary': {
                'total_transactions': len(df),
                'vendor_standardizations': self.stats['vendor_standardizations'],
                'category_classifications': self.stats['category_classifications'],
                'transaction_intelligence': self.stats['transaction_intelligence'],
                'llm_calls': combined_llm_calls,
                'cache_hit_rate': self.cache.get_cache_stats()['vendor_hit_rate']
            },
            'cost_analysis': combined_cost,
            'performance_metrics': combined_perf,
            'cache_performance': self.cache.get_cache_stats(),
            'quality_metrics': self._calculate_quality_metrics(processing_results)
        }
        # Back-compat tracker summary for external tests
        summary_report['llm_tracker'] = {
            'total_calls': combined_llm_calls,
            'total_cost': combined_cost.get('total_cost', 0.0)
        }
        
        # Audit Logs
        audit_logs = {
            'processing_timeline': processing_results,
            'llm_requests': self._generate_llm_audit_log(),
            'cache_operations': self._generate_cache_audit_log(),
            'error_logs': self._generate_error_log()
        }
        
        # Transaction Intelligence Summary
        transaction_intelligence = self._summarize_transaction_intelligence(processing_results)
        
        return {
            'summary_report': summary_report,
            'audit_logs': audit_logs,
            'transaction_intelligence': transaction_intelligence,
            'schema_analysis': schema_analysis
        }

    # Helper methods for column detection
    def _find_vendor_columns(self, df: pd.DataFrame) -> List[str]:
        """Find vendor-related columns (prioritize standardized/clean fields)."""
        vendor_keywords = ['standardized_vendor', 'clean vendor', 'clean_vendor', 'merchant', 'vendor', 'store', 'business', 'company', 'payee', 'name']
        try:
            return [col for col in df.columns if any(keyword in str(col).lower() for keyword in vendor_keywords)]
        except Exception as e:
            logger.error(f"Error in _find_vendor_columns: {e}")
            logger.error(f"DataFrame columns: {list(df.columns)}")
            logger.error(f"Column types: {[type(col) for col in df.columns]}")
            return []
    
    def _find_amount_columns(self, df: pd.DataFrame) -> List[str]:
        """Find amount-related columns"""
        amount_keywords = ['amount', 'price', 'cost', 'value', 'total']
        try:
            return [col for col in df.columns if any(keyword in str(col).lower() for keyword in amount_keywords)]
        except Exception as e:
            logger.error(f"Error in _find_amount_columns: {e}")
            return []
    
    def _find_category_columns(self, df: pd.DataFrame) -> List[str]:
        """Find category-related columns"""
        category_keywords = ['category', 'type', 'classification', 'group']
        try:
            return [col for col in df.columns if any(keyword in str(col).lower() for keyword in category_keywords)]
        except Exception as e:
            logger.error(f"Error in _find_category_columns: {e}")
            return []
    
    def _find_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Find date-related columns"""
        date_keywords = ['date', 'time', 'timestamp']
        try:
            return [col for col in df.columns if any(keyword in str(col).lower() for keyword in date_keywords)]
        except Exception as e:
            logger.error(f"Error in _find_date_columns: {e}")
            return []
    
    def _find_description_columns(self, df: pd.DataFrame) -> List[str]:
        """Find description-related columns"""
        description_keywords = ['description', 'desc', 'note', 'details', 'comment']
        try:
            return [col for col in df.columns if any(keyword in str(col).lower() for keyword in description_keywords)]
        except Exception as e:
            logger.error(f"Error in _find_description_columns: {e}")
            return []
    
    def _assess_processing_capabilities(self, column_analysis: Dict[str, List[str]]) -> Dict[str, bool]:
        """Assess what processing capabilities are available"""
        return {
            'vendor_standardization': len(column_analysis['vendor_columns']) > 0,
            'category_classification': len(column_analysis['vendor_columns']) > 0 and len(column_analysis['amount_columns']) > 0,
            'amount_processing': len(column_analysis['amount_columns']) > 0,
            'date_processing': len(column_analysis['date_columns']) > 0,
            'description_enhancement': len(column_analysis['description_columns']) > 0
        }

    # Rule-based processing methods
    def _apply_vendor_rules(self, vendor: str) -> Dict[str, Any]:
        """Apply rule-based vendor standardization"""
        vendor_lower = str(vendor).lower()
        
        # If force_llm_for_testing is enabled, skip rules for some vendors to force LLM calls
        if self.config.get('force_llm_for_testing', False):
            # Only apply rules to very common vendors, let others go to LLM
            limited_mappings = {
                'google': 'Google Workspace',
                'netflix': 'Netflix',
                'uber': 'Uber',
                'lyft': 'Lyft'
            }
            
            for pattern, standardized in limited_mappings.items():
                if pattern in vendor_lower:
                    return {
                        'matched': True,
                        'standardized': standardized,
                        'confidence': 0.95,
                        'explanation': f"Rule-based mapping: {pattern} → {standardized}"
                    }
            
            return {'matched': False}
        
        # Enhanced vendor pattern matching with real-world formats
        vendor_mappings = {
            # Google variants
            'google': 'Google Workspace', 'googl': 'Google Workspace', 'goog': 'Google Workspace',
            'google workspace': 'Google Workspace', 'google ads': 'Google Ads', 
            'google cloud': 'Google Cloud', 'youtube': 'YouTube',
            
            # Meta/Facebook variants
            'meta': 'Meta', 'facebook': 'Meta', 'fb ': 'Meta', 'meta platforms': 'Meta',
            'instagram': 'Meta', 'whatsapp': 'Meta',
            
            # Amazon variants
            'amazon': 'Amazon', 'amzn': 'Amazon', 'aws': 'Amazon Web Services',
            'amazon web services': 'Amazon Web Services', 'amazon prime': 'Amazon Prime',
            'amzn mktp': 'Amazon', 'amazon.com': 'Amazon',
            
            # Payment processors (common patterns)
            'stripe': 'Stripe', 'stripe*': 'Stripe', 'stripe inc': 'Stripe',
            'paypal': 'PayPal', 'paypal *': 'PayPal', 'pp*': 'PayPal',
            'square': 'Square', 'sq *': 'Square',
            
            # Streaming services
            'netflix': 'Netflix', 'netflix.com': 'Netflix', 'nflx': 'Netflix',
            'spotify': 'Spotify', 'spotify premium': 'Spotify', 'spotify usa': 'Spotify',
            'hulu': 'Hulu', 'disney+': 'Disney Plus', 'apple music': 'Apple Music',
            
            # Transportation
            'uber': 'Uber', 'uber *': 'Uber', 'uber eats': 'Uber Eats',
            'lyft': 'Lyft', 'lyft *': 'Lyft',
            'delta': 'Delta Airlines', 'united': 'United Airlines', 'southwest': 'Southwest Airlines',
            
            # Technology/Cloud services
            'digitalocean': 'DigitalOcean', 'digital ocean': 'DigitalOcean',
            'github': 'GitHub', 'slack': 'Slack', 'zoom': 'Zoom',
            'microsoft': 'Microsoft', 'msft': 'Microsoft', 'office 365': 'Microsoft 365',
            'adobe': 'Adobe', 'salesforce': 'Salesforce', 'dropbox': 'Dropbox',
            
            # Food/Coffee
            'starbucks': 'Starbucks', 'sbux': 'Starbucks', 'starbux': 'Starbucks',
            'mcdonalds': 'McDonalds', 'mcd': 'McDonalds', 'chipotle': 'Chipotle',
            'doordash': 'DoorDash', 'grubhub': 'GrubHub', 'seamless': 'Seamless',
            
            # Banking/Finance
            'chase': 'JPMorgan Chase', 'wells fargo': 'Wells Fargo', 'bank of america': 'Bank of America',
            'american express': 'American Express', 'amex': 'American Express',
            'visa': 'Visa', 'mastercard': 'Mastercard'
        }
        
        for pattern, standardized in vendor_mappings.items():
            if pattern in vendor_lower:
                return {
                    'matched': True,
                    'standardized': standardized,
                    'confidence': 0.95,
                    'explanation': f"Rule-based mapping: {pattern} → {standardized}"
                }
        
        return {'matched': False}
    
    def _apply_category_rules(self, vendor: str, amount: float) -> Dict[str, Any]:
        """Apply rule-based category classification with amount-based hints"""
        # Handle case where vendor might be NaN or not a string
        if vendor is None or pd.isna(vendor) or not isinstance(vendor, str):
            return {'matched': False}
        
        vendor_lower = vendor.lower()
        
        # Debug logging to see what vendors we're getting
        logger.debug(f"Category rules processing vendor: '{vendor}' (${amount})")
        
        # Amount-based category hints (applied first for better accuracy)
        if amount <= 15:  # Small amounts often indicate meals/coffee
            if any(keyword in vendor_lower for keyword in ['coffee', 'cafe', 'starbucks', 'sbux', 'dunkin']):
                return {
                    'matched': True,
                    'category': 'Meals & Entertainment',
                    'confidence': 0.9,
                    'explanation': f"Small amount (${amount}) + coffee vendor = Meals & Entertainment"
                }
        
        if 5 <= amount <= 50:  # Meal range
            if any(keyword in vendor_lower for keyword in ['restaurant', 'food', 'pizza', 'burger', 'mcd']):
                return {
                    'matched': True,
                    'category': 'Meals & Entertainment',
                    'confidence': 0.85,
                    'explanation': f"Meal amount range (${amount}) + food vendor = Meals & Entertainment"
                }
        
        if amount >= 500:  # Large amounts often indicate business services
            if any(keyword in vendor_lower for keyword in ['consulting', 'services', 'professional']):
                return {
                    'matched': True,
                    'category': 'Professional Services',
                    'confidence': 0.8,
                    'explanation': f"Large amount (${amount}) + service vendor = Professional Services"
                }
        
        # Enhanced vendor-specific rules with proper categories
        category_rules = [
            # Software & Technology
            ('google', 'Software & Technology'),
            ('microsoft', 'Software & Technology'),
            ('adobe', 'Software & Technology'),
            ('github', 'Software & Technology'),
            ('digitalocean', 'Software & Technology'),
            ('aws', 'Software & Technology'),
            
            # Marketing & Advertising  
            ('meta', 'Marketing & Advertising'),
            ('facebook', 'Marketing & Advertising'),
            ('google ads', 'Marketing & Advertising'),
            
            # Meals & Entertainment
            ('netflix', 'Meals & Entertainment'),
            ('spotify', 'Meals & Entertainment'),
            ('starbucks', 'Meals & Entertainment'),
            ('mcdonalds', 'Meals & Entertainment'),
            
            # Travel & Transportation
            ('uber', 'Travel & Transportation'),
            ('lyft', 'Travel & Transportation'),
            ('delta', 'Travel & Transportation'),
            ('united', 'Travel & Transportation'),
            
            # Banking & Finance
            ('stripe', 'Banking & Finance'),
            ('paypal', 'Banking & Finance'),
            ('square', 'Banking & Finance'),
            ('chase', 'Banking & Finance'),
            
            # Office Supplies & Equipment (context-dependent)
            ('amazon', 'Office Supplies & Equipment'),
            ('staples', 'Office Supplies & Equipment')
        ]
        
        for pattern, category in category_rules:
            if pattern in vendor_lower:
                return {
                    'matched': True,
                    'category': category,
                    'confidence': 0.9,
                    'explanation': f"Rule-based category: {pattern} → {category}"
                }
        
        return {'matched': False}

    # Transaction intelligence methods
    def _generate_transaction_tags(self, row: pd.Series) -> List[str]:
        """Generate transaction tags"""
        tags = []
        
        amount = float(row.get('amount', 0))
        if amount > 1000:
            tags.append('high_value')
        elif amount < 10:
            tags.append('low_value')
        
        vendor = str(row.get('merchant', '') or '').lower()
        if any(keyword in vendor for keyword in ['subscription', 'monthly', 'annual']):
            tags.append('subscription')
        if any(keyword in vendor for keyword in ['food', 'restaurant', 'cafe']):
            tags.append('food_dining')
        
        return tags
    
    def _generate_transaction_insights(self, row: pd.Series) -> List[str]:
        """Generate transaction insights"""
        insights = []
        
        amount = float(row.get('amount', 0))
        vendor = str(row.get('merchant', ''))
        
        if amount > 500:
            insights.append(f"High-value transaction: ${amount:.2f} at {vendor}")
        
        if 'subscription' in vendor.lower():
            insights.append("Recurring subscription payment detected")
        
        return insights
    
    def _generate_transaction_explanation(self, row: pd.Series) -> str:
        """Generate transaction explainability"""
        amount = float(row.get('amount', 0))
        vendor = str(row.get('merchant', ''))
        
        if amount > 1000:
            return f"High-value transaction requiring attention"
        elif amount < 10:
            return f"Low-value transaction, likely incidental"
        else:
            return f"Standard transaction amount"
    
    def _calculate_transaction_risk(self, row: pd.Series) -> float:
        """Calculate transaction risk score (0-1)"""
        risk_score = 0.0
        
        amount = float(row.get('amount', 0))
        if amount > 1000:
            risk_score += 0.3
        if amount > 5000:
            risk_score += 0.4
        
        vendor = str(row.get('merchant', '') or '').lower()
        if any(keyword in vendor for keyword in ['unknown', 'unrecognized']):
            risk_score += 0.2
        
        return min(1.0, risk_score)
    
    def _detect_transaction_anomalies(self, row: pd.Series) -> Dict[str, Any]:
        """Detect transaction anomalies"""
        anomalies = {}
        
        amount = float(row.get('amount', 0))
        vendor = str(row.get('merchant', ''))
        
        if amount > 10000:
            anomalies['high_amount'] = True
        if amount < 0:
            anomalies['negative_amount'] = True
        
        if 'unknown' in vendor.lower() or vendor.strip() == '':
            anomalies['unknown_vendor'] = True
        
        return anomalies

    # Output generation methods
    def _calculate_quality_metrics(self, processing_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate quality metrics from processing results"""
        vendor_confidences = []
        category_confidences = []
        
        for result in processing_results:
            if result['vendor_standardization']:
                vendor_confidences.append(result['vendor_standardization']['confidence'])
            if result['category_classification']:
                category_confidences.append(result['category_classification']['confidence'])
        
        return {
            'average_vendor_confidence': np.mean(vendor_confidences) if vendor_confidences else 0,
            'average_category_confidence': np.mean(category_confidences) if category_confidences else 0,
            'vendor_standardizations': len(vendor_confidences),
            'category_classifications': len(category_confidences)
        }
    
    def _summarize_transaction_intelligence(self, processing_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize transaction intelligence across all rows"""
        all_tags = []
        all_insights = []
        all_risk_scores = []
        all_anomalies = []
        
        for result in processing_results:
            if result['transaction_intelligence']:
                intelligence = result['transaction_intelligence']
                all_tags.extend(intelligence['tags'])
                all_insights.extend(intelligence['insights'])
                all_risk_scores.append(intelligence['risk_score'])
                all_anomalies.append(intelligence['anomaly_detection'])
        
        return {
            'tags_summary': list(set(all_tags)),
            'insights_summary': all_insights,
            'average_risk_score': np.mean(all_risk_scores) if all_risk_scores else 0,
            'anomaly_report': all_anomalies
        }
    
    def _generate_llm_audit_log(self) -> List[Dict[str, Any]]:
        """Generate LLM audit log"""
        return {
            'total_calls': self.tracker.performance_tracker['total_llm_calls'],
            'successful_calls': self.tracker.performance_tracker['successful_llm_calls'],
            'failed_calls': self.tracker.performance_tracker['failed_llm_calls'],
            'average_response_time': self.tracker.performance_tracker['average_response_time']
        }
    
    def _generate_cache_audit_log(self) -> Dict[str, Any]:
        """Generate cache audit log"""
        return self.cache.get_cache_stats()
    
    def _generate_error_log(self) -> List[str]:
        """Generate error log"""
        return []  # Would be populated with actual errors
    
    # Legacy column finding methods for compatibility
    def _find_vendor_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the best column to use as vendor, prefer standardized/clean vendor fields."""
        # Exact-priority candidates
        priority_exact = ['standardized_vendor', 'clean vendor', 'clean_vendor']
        for cand in priority_exact:
            for col in df.columns:
                if str(col).lower() == cand:
                    logger.info(f"Selected vendor column (priority): {col}")
                    return col

        # Fallback to common vendor-like columns by substring
        vendor_keywords = ['merchant', 'vendor', 'store', 'business', 'company', 'payee', 'name']
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in vendor_keywords):
                logger.info(f"Selected vendor column (fallback): {col}")
                return col
        
        logger.info("No vendor column detected")
        return None
    
    def _find_amount_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the best column to use as amount"""
        amount_keywords = ['amount', 'price', 'cost', 'value']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in amount_keywords):
                return col
        
        # Try to find numeric columns
        for col in df.columns:
            if col.lower() not in ['merchant', 'vendor', 'store', 'description', 'date', 'time', 'id']:
                numeric_count = 0
                for val in df[col]:
                    try:
                        float(str(val).replace('$', '').replace(',', ''))
                        numeric_count += 1
                    except (ValueError, TypeError):
                        pass
                
                if numeric_count > len(df[col]) * 0.5:
                    return col
        
        return None

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        return {
            **self.stats,
            'cache_stats': self.cache.get_cache_stats(),
            'tracking_summary': self.tracker.get_tracking_summary()
        } 