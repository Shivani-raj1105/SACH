@echo off
echo 🚀 Preparing files for deployment...
echo.

echo 📁 Creating deployment folder...
if not exist "deployment" mkdir deployment
if not exist "deployment\saved_model" mkdir deployment\saved_model

echo 📋 Copying files...
copy "streamlit_app.py" "deployment\"
copy "requirements_streamlit.txt" "deployment\"
copy "saved_model\*.*" "deployment\saved_model\"

echo.
echo ✅ Files ready for deployment!
echo.
echo 📂 Your deployment folder is: deployment\
echo.
echo 📋 Files to upload:
echo    - streamlit_app.py
echo    - requirements_streamlit.txt
echo    - saved_model\ (all files)
echo.
echo 🚀 Next steps:
echo    1. Go to https://share.streamlit.io
echo    2. Sign in with GitHub
echo    3. Click "New app"
echo    4. Upload files from deployment\ folder
echo    5. Set main file to: streamlit_app.py
echo    6. Deploy!
echo.
pause 