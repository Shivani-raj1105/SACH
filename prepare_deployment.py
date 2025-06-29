#!/usr/bin/env python3
"""
Deployment Preparation Script for Fake News Detector
This script helps you prepare your app for deployment to share with friends.
"""

import os
import sys
import re

def check_files():
    """Check if all required files exist"""
    print("🔍 Checking required files...")
    
    required_files = [
        "streamlit_app.py",
        "requirements_streamlit.txt",
        "saved_model/config.json",
        "saved_model/model.safetensors", 
        "saved_model/vocab.txt",
        "saved_model/tokenizer_config.json",
        "saved_model/special_tokens_map.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("\n✅ All required files found!")
        return True

def update_contact_info():
    """Help user update contact information in the app"""
    print("\n📝 Contact Information Update")
    print("=" * 40)
    
    # Read current app file
    try:
        with open("streamlit_app.py", "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ streamlit_app.py not found!")
        return False
    
    # Check current contact info
    current_name = re.search(r'<b>Created by:</b> ([^<]+)', content)
    current_email = re.search(r'<b>Contact:</b> ([^<]+)', content)
    
    print(f"Current name: {current_name.group(1) if current_name else 'Not set'}")
    print(f"Current email: {current_email.group(1) if current_email else 'Not set'}")
    
    # Get new info
    print("\nEnter your information (or press Enter to keep current):")
    new_name = input("Your name: ").strip()
    new_email = input("Your email: ").strip()
    
    # Update if provided
    if new_name:
        content = re.sub(r'(<b>Created by:</b> )[^<]+', r'\1' + new_name, content)
    if new_email:
        content = re.sub(r'(<b>Contact:</b> )[^<]+', r'\1' + new_email, content)
    
    # Write back
    try:
        with open("streamlit_app.py", "w", encoding="utf-8") as f:
            f.write(content)
        print("✅ Contact information updated!")
        return True
    except Exception as e:
        print(f"❌ Error updating file: {e}")
        return False

def check_model_size():
    """Check model file sizes"""
    print("\n📊 Model File Sizes:")
    print("=" * 40)
    
    model_files = [
        "saved_model/config.json",
        "saved_model/model.safetensors",
        "saved_model/vocab.txt",
        "saved_model/tokenizer_config.json",
        "saved_model/special_tokens_map.json"
    ]
    
    total_size = 0
    for file_path in model_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            total_size += size
            size_mb = size / (1024 * 1024)
            print(f"📁 {file_path}: {size_mb:.1f} MB")
    
    total_mb = total_size / (1024 * 1024)
    print(f"\n📦 Total model size: {total_mb:.1f} MB")
    
    if total_mb > 500:
        print("⚠️  Warning: Large model size may cause slow deployment")
    else:
        print("✅ Model size is reasonable for deployment")

def create_deployment_package():
    """Create a deployment package checklist"""
    print("\n📦 Deployment Package Checklist:")
    print("=" * 40)
    
    checklist = [
        "✅ All model files present",
        "✅ streamlit_app.py ready",
        "✅ requirements_streamlit.txt ready", 
        "✅ Contact info updated",
        "✅ Tested locally",
        "📝 Choose deployment platform:",
        "   - Streamlit Cloud (Recommended)",
        "   - Hugging Face Spaces",
        "   - Railway",
        "   - Local network sharing",
        "📝 Create GitHub repository (if using cloud platforms)",
        "📝 Upload all files to repository",
        "📝 Deploy using platform instructions",
        "📝 Share URL with friends!"
    ]
    
    for item in checklist:
        print(f"   {item}")

def main():
    """Main deployment preparation function"""
    print("🚀 Fake News Detector - Deployment Preparation")
    print("=" * 50)
    
    # Check files
    files_ok = check_files()
    if not files_ok:
        print("\n❌ Please ensure all required files are present before deploying.")
        return
    
    # Check model size
    check_model_size()
    
    # Update contact info
    update_contact_info()
    
    # Show deployment checklist
    create_deployment_package()
    
    print("\n🎉 Deployment preparation complete!")
    print("\nNext steps:")
    print("1. Read DEPLOYMENT_GUIDE.md for detailed instructions")
    print("2. Choose your deployment platform")
    print("3. Follow the platform-specific instructions")
    print("4. Share your app URL with friends!")
    
    print("\n💡 Quick start recommendation:")
    print("   Go to https://share.streamlit.io and deploy using streamlit_app.py")

if __name__ == "__main__":
    main() 