#!/usr/bin/env python3

import pandas as pd
import sys
sys.path.append('.')
from production_cleaner_ai_v5 import AIEnhancedProductionCleanerV5
from llm_client_v2 import LLMClient

# Quick test with 5 rows
test_data = [
    {'merchant': 'Google LLC', 'amount': 15.0, 'description': 'Cloud storage'},
    {'merchant': 'Unknown Corp', 'amount': 50.0, 'description': 'Service'},
    {'merchant': 'Netflix Inc', 'amount': 12.99, 'description': 'Streaming'},
    {'merchant': 'STRIPE*PAYMENT', 'amount': 99.0, 'description': 'Software'},
    {'merchant': 'Test Vendor', 'amount': 25.0, 'description': 'Testing'}
]

print("🧪 Quick Parallel Processing Test")
print("=" * 40)

df = pd.DataFrame(test_data)
config = {
    'enable_ai': True,
    'ai_vendor_enabled': True,
    'ai_category_enabled': True,
    'enable_parallel_processing': True,
    'max_workers': 2,
    'enable_source_tracking': True,
    'force_llm_for_testing': False
}

llm_client = LLMClient(use_mock=True, enable_caching=True)
processor = AIEnhancedProductionCleanerV5(df=df, config=config, llm_client=llm_client)

try:
    cleaned_df, result = processor.process_data()
    print(f'✅ Processed {len(cleaned_df)} rows successfully!')
    
    # Check if we have the expected columns
    print(f"📊 Columns: {list(cleaned_df.columns)}")
    
    # Print some results
    if 'standardized_vendor' in cleaned_df.columns:
        print("🏢 Vendor Results:")
        for i, row in cleaned_df.iterrows():
            original = row.get('merchant', 'N/A')
            standardized = row.get('standardized_vendor', 'N/A')
            print(f"   {original} -> {standardized}")
    
    # Check summary
    summary = result.get('summary_report', {})
    print(f"\n📈 Summary:")
    print(f"   Processing time: {summary.get('processing_summary', {}).get('total_time', 'N/A')}s")
    print(f"   LLM calls: {summary.get('processing_summary', {}).get('llm_calls', 'N/A')}")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc() 