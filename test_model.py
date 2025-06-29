#!/usr/bin/env python3
"""
Simple test script to check if the model loads and works properly.
"""

import time
import sys

def test_model_loading():
    print("🧠 Testing model loading...")
    start_time = time.time()
    
    try:
        # Test model import
        print("📦 Importing model...")
        from model import predict_news
        print(f"✅ Model imported successfully in {time.time() - start_time:.1f}s")
        
        # Test prediction
        print("🔍 Testing prediction...")
        test_text = "This is a test news article for fake news detection."
        pred_start = time.time()
        
        label, confidence, explanation = predict_news(test_text)
        
        print(f"✅ Prediction completed in {time.time() - pred_start:.1f}s")
        print(f"📊 Result: {label} (confidence: {confidence:.2%})")
        print(f"🔍 Top words: {explanation[:3]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoint():
    print("\n🌐 Testing API endpoint...")
    try:
        import requests
        
        # Test if server is running
        try:
            response = requests.get("http://127.0.0.1:8000/docs", timeout=5)
            print("✅ API server is running")
        except:
            print("❌ API server is not running on http://127.0.0.1:8000")
            return False
        
        # Test prediction endpoint
        test_data = {"text": "This is a test news article."}
        start_time = time.time()
        
        response = requests.post(
            "http://127.0.0.1:8000/predict", 
            json=test_data, 
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API prediction successful in {time.time() - start_time:.1f}s")
            print(f"📊 Result: {result}")
            return True
        else:
            print(f"❌ API error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🧠 Fake News Detector - Model Test")
    print("=" * 50)
    
    # Test model loading
    model_ok = test_model_loading()
    
    # Test API endpoint
    api_ok = test_api_endpoint()
    
    print("\n" + "=" * 50)
    print("📋 Test Results:")
    print(f"Model Loading: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"API Endpoint: {'✅ PASS' if api_ok else '❌ FAIL'}")
    print("=" * 50)
    
    if model_ok and api_ok:
        print("🎉 Everything is working! Your app should be ready to use.")
    else:
        print("⚠️  Some tests failed. Check the errors above.") 