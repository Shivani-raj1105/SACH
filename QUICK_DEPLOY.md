# 🚀 Quick Deploy Guide (No Git Required)

## Step 1: Create GitHub Account
1. Go to [github.com](https://github.com)
2. Click "Sign up"
3. Create your account (free)

## Step 2: Create Repository
1. Click the "+" icon in top right
2. Select "New repository"
3. Name it: `fake-news-detector`
4. Make it **Public**
5. Click "Create repository"

## Step 3: Upload Files
1. In your new repository, click "uploading an existing file"
2. **Drag and drop these files:**
   ```
   streamlit_app.py
   requirements_streamlit.txt
   saved_model/config.json
   saved_model/model.safetensors
   saved_model/vocab.txt
   saved_model/tokenizer_config.json
   saved_model/special_tokens_map.json
   ```
3. Click "Commit changes"

## Step 4: Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. **Repository:** Select your `fake-news-detector` repo
5. **Branch:** main
6. **Main file path:** `streamlit_app.py`
7. **Requirements file:** `requirements_streamlit.txt`
8. Click "Deploy"

## Step 5: Share with Friends
Your app will be available at:
```
https://fake-news-detector-yourusername.streamlit.app
```

## 🎉 Done!
Share this URL with your friends!

---

## Alternative: Manual Upload (Even Easier)

If GitHub seems complicated:

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Choose "Upload files"
5. Upload all your files
6. Set main file to `streamlit_app.py`
7. Deploy!

## 🆘 Need Help?

- **GitHub Help:** [help.github.com](https://help.github.com)
- **Streamlit Cloud:** [docs.streamlit.io](https://docs.streamlit.io)
- **Contact:** shivani.raj.urs1105@gmail.com 