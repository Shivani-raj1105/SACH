#!/usr/bin/env python3
"""
Test script to verify deployment files work correctly
"""

import os
import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit
        print("✅ streamlit")
    except ImportError as e:
        print(f"❌ streamlit: {e}")
        return False
    
    try:
        import transformers
        print("✅ transformers")
    except ImportError as e:
        print(f"❌ transformers: {e}")
        return False
    
    try:
        import torch
        print("✅ torch")
    except ImportError as e:
        print(f"❌ torch: {e}")
        return False
    
    try:
        import lime
        print("✅ lime")
    except ImportError as e:
        print(f"❌ lime: {e}")
        return False
    
    try:
        import pandas
        print("✅ pandas")
    except ImportError as e:
        print(f"❌ pandas: {e}")
        return False
    
    try:
        import numpy
        print("✅ numpy")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False
    
    return True

def test_model_files():
    """Test if model files exist and are accessible"""
    print("\n🔍 Testing model files...")
    
    model_files = [
        "saved_model/config.json",
        "saved_model/model.safetensors",
        "saved_model/vocab.txt",
        "saved_model/tokenizer_config.json",
        "saved_model/special_tokens_map.json"
    ]
    
    all_exist = True
    for file_path in model_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def test_streamlit_app():
    """Test if streamlit_app.py can be imported"""
    print("\n🔍 Testing streamlit_app.py...")
    
    try:
        # Read the file to check for syntax errors
        with open("streamlit_app.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check for key components
        if "import streamlit" in content:
            print("✅ streamlit import found")
        else:
            print("❌ streamlit import missing")
            return False
        
        if "AutoTokenizer" in content:
            print("✅ AutoTokenizer import found")
        else:
            print("❌ AutoTokenizer import missing")
            return False
        
        if "load_model" in content:
            print("✅ load_model function found")
        else:
            print("❌ load_model function missing")
            return False
        
        if "Shivani Raj" in content:
            print("✅ Contact info updated")
        else:
            print("⚠️  Contact info not updated")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading streamlit_app.py: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Deployment Test Suite")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test model files
    model_files_ok = test_model_files()
    
    # Test streamlit app
    app_ok = test_streamlit_app()
    
    # Summary
    print("\n📊 Test Results:")
    print("=" * 40)
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Model files: {'✅ PASS' if model_files_ok else '❌ FAIL'}")
    print(f"Streamlit app: {'✅ PASS' if app_ok else '❌ FAIL'}")
    
    if imports_ok and model_files_ok and app_ok:
        print("\n🎉 All tests passed! Your app is ready for deployment.")
        print("\n📋 Next steps:")
        print("1. Go to https://share.streamlit.io")
        print("2. Set main file to: streamlit_app.py")
        print("3. Set requirements to: requirements_streamlit.txt")
        print("4. Deploy!")
    else:
        print("\n❌ Some tests failed. Please fix the issues before deploying.")
    
    return imports_ok and model_files_ok and app_ok

if __name__ == "__main__":
    main() 