#!/usr/bin/env python3
"""
Startup script for the Fake News Detector application.
This script will start the FastAPI server and provide instructions for running the Streamlit app.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        'transformers', 'torch', 'fastapi', 'uvicorn', 
        'streamlit', 'requests', 'lime', 'pandas'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_model_files():
    """Check if the saved model files exist."""
    required_files = [
        'saved_model/config.json',
        'saved_model/model.safetensors', 
        'saved_model/vocab.txt',
        'saved_model/tokenizer_config.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing model files: {', '.join(missing_files)}")
        return False
    
    print("✅ All model files are present!")
    return True

def start_api_server():
    """Start the FastAPI server."""
    print("🚀 Starting FastAPI server...")
    try:
        # Start the API server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "api:app", 
            "--host", "127.0.0.1", "--port", "8000", "--reload"
        ])
        
        print("✅ FastAPI server started successfully!")
        print("📡 API endpoint: http://127.0.0.1:8000")
        print("📖 API docs: http://127.0.0.1:8000/docs")
        
        return process
        
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return None

def main():
    print("=" * 60)
    print("🧠 Fake News Detector - Startup Script")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check model files
    if not check_model_files():
        return
    
    print("\n" + "=" * 60)
    print("🎯 Starting the application...")
    print("=" * 60)
    
    # Start API server
    api_process = start_api_server()
    
    if api_process:
        print("\n" + "=" * 60)
        print("📱 To start the Streamlit web interface:")
        print("=" * 60)
        print("Open a NEW terminal window and run:")
        print("streamlit run app.py")
        print("\n🌐 The web app will be available at: http://localhost:8501")
        
        print("\n" + "=" * 60)
        print("🛑 To stop the application:")
        print("=" * 60)
        print("Press Ctrl+C in this terminal to stop the API server")
        print("Press Ctrl+C in the Streamlit terminal to stop the web app")
        
        try:
            # Keep the API server running
            api_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping API server...")
            api_process.terminate()
            print("✅ API server stopped.")

if __name__ == "__main__":
    main() 