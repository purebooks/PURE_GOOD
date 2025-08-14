#!/usr/bin/env python3
"""
Test script for the simple test interface
"""

import requests
import json

def test_simple_interface():
    """Test the simple interface functionality"""
    
    print("🧪 Testing Simple Interface with API")
    print("=" * 50)
    
    # Test data similar to what the interface would send
    test_data = {
        "user_intent": "find duplicates and errors",
        "data": {
            "merchant": [
                "PAYPAL*DIGITALOCEAN",
                "SQ *COFFEE SHOP NYC",
                "UBER EATS DEC15",
                "PAYPAL*DIGITALOCEAN",  # Duplicate
                "AMAZON.COM*AMZN.COM/BILL"
            ],
            "amount": [50.00, 4.50, 23.75, 50.00, 12.99],
            "description": [
                "DigitalOcean hosting",
                "Coffee purchase",
                "Food delivery",
                "DigitalOcean hosting",  # Duplicate
                "Amazon Prime"
            ]
        }
    }
    
    try:
        print("📤 Sending test data to API...")
        response = requests.post(
            'http://localhost:8080/process',
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful!")
            print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
            print(f"   Rows processed: {len(result.get('cleaned_data', []))}")
            print(f"   AI requests: {result.get('insights', {}).get('ai_requests', 0)}")
            print(f"   AI cost: ${result.get('insights', {}).get('ai_cost', 0):.3f}")
            
            # Show some cleaned data
            cleaned_data = result.get('cleaned_data', [])
            if cleaned_data:
                print("\n📊 Sample cleaned data:")
                for i, row in enumerate(cleaned_data[:3]):
                    print(f"   {i+1}. {row}")
            
            # Show vendor transformations if any
            transformations = result.get('insights', {}).get('vendor_transformations', [])
            if transformations:
                print("\n🔄 Vendor transformations:")
                for transform in transformations:
                    print(f"   {transform}")
            
            return True
            
        else:
            print(f"❌ API call failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_interface()
    if success:
        print("\n🎉 Simple interface test passed!")
    else:
        print("\n❌ Simple interface test failed!") 