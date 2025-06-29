# 🚀 Deploy Your Fake News Detector for Friends

This guide will help you deploy your fake news detector so your friends can use it online!

## Option 1: Streamlit Cloud (Recommended - FREE) ⭐

**Best for:** Sharing with friends quickly and easily
**Cost:** Free
**Setup time:** 10-15 minutes

### Step 1: Prepare Your Files

1. **Ensure you have these files in your project:**
   ```
   fake_news/
   ├── streamlit_app.py          # Main app file
   ├── requirements_streamlit.txt # Dependencies
   ├── saved_model/              # Your trained model
   │   ├── config.json
   │   ├── model.safetensors
   │   ├── vocab.txt
   │   ├── tokenizer_config.json
   │   └── special_tokens_map.json
   └── README.md
   ```

2. **Update your contact info in `streamlit_app.py`:**
   ```python
   # Find these lines and update them:
   <b>Created by:</b> Your Name<br>
   <b>Contact:</b> your.email@example.com
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in with GitHub** (create account if needed)
3. **Click "New app"**
4. **Connect your GitHub repository:**
   - If your code is on GitHub, select your repository
   - If not, upload your files or create a GitHub repo first
5. **Configure the app:**
   - **Main file path:** `streamlit_app.py`
   - **Python version:** 3.9 or 3.10
   - **Requirements file:** `requirements_streamlit.txt`
6. **Click "Deploy"**

### Step 3: Share with Friends

Once deployed, you'll get a URL like:
```
https://your-app-name-your-username.streamlit.app
```

Share this URL with your friends! 🎉

---

## Option 2: Hugging Face Spaces (Alternative - FREE)

**Best for:** More advanced features and customization
**Cost:** Free
**Setup time:** 15-20 minutes

### Step 1: Create a Hugging Face Account

1. Go to [huggingface.co](https://huggingface.co)
2. Sign up for a free account

### Step 2: Create a New Space

1. Click "New Space"
2. Choose "Streamlit" as the SDK
3. Give it a name like "fake-news-detector"
4. Set to "Public" so friends can access it

### Step 3: Upload Your Files

Upload these files to your Space:
- `streamlit_app.py`
- `requirements_streamlit.txt`
- All files from `saved_model/` folder

### Step 4: Share the URL

Your app will be available at:
```
https://huggingface.co/spaces/your-username/fake-news-detector
```

---

## Option 3: Railway (Paid but Powerful)

**Best for:** High performance and custom domains
**Cost:** $5-20/month
**Setup time:** 20-30 minutes

### Step 1: Create Railway Account

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub

### Step 2: Deploy Your App

1. Click "New Project"
2. Choose "Deploy from GitHub repo"
3. Select your repository
4. Railway will auto-detect it's a Python app

### Step 3: Configure Environment

Add these environment variables:
```
PYTHON_VERSION=3.9
```

### Step 4: Deploy and Share

Railway will give you a URL like:
```
https://your-app-name.railway.app
```

---

## Option 4: Local Network Sharing (Free)

**Best for:** Sharing on the same WiFi network
**Cost:** Free
**Setup time:** 5 minutes

### Step 1: Find Your IP Address

**Windows:**
```bash
ipconfig
```
Look for "IPv4 Address" (usually 192.168.x.x)

**Mac/Linux:**
```bash
ifconfig
```

### Step 2: Run Your App

```bash
# Start the API server
uvicorn api:app --host 0.0.0.0 --port 8000

# In another terminal, start Streamlit
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### Step 3: Share the URL

Your friends on the same WiFi can access:
```
http://YOUR_IP_ADDRESS:8501
```

---

## 🎯 Quick Start Checklist

Before deploying, make sure you have:

- [ ] All model files in `saved_model/` folder
- [ ] Updated contact info in the app
- [ ] Tested the app locally
- [ ] Chosen a deployment option
- [ ] Created necessary accounts (GitHub, etc.)

## 🔧 Troubleshooting

### Common Issues:

1. **"Model not found" error:**
   - Ensure all files in `saved_model/` are uploaded
   - Check file permissions

2. **"Requirements not found":**
   - Make sure `requirements_streamlit.txt` is in the root directory
   - Check for typos in package names

3. **"App won't load":**
   - Check the logs in your deployment platform
   - Ensure Python version is compatible (3.8-3.10)

4. **"Slow loading":**
   - First load is always slow (model downloading)
   - Subsequent loads will be faster

### Performance Tips:

- The model is ~500MB, so first deployment takes time
- Consider using a smaller model for faster loading
- Enable caching in Streamlit for better performance

## 📱 Mobile Access

All deployment options work on mobile devices! Your friends can:
- Access the app on their phones
- Use it on tablets
- Share results via social media

## 🎉 Success!

Once deployed, your friends can:
1. Visit your app URL
2. Paste any news article or headline
3. Get instant fake news detection
4. See confidence scores and explanations
5. Share results with others

**Your app is now accessible to the world! 🌍**

---

## Need Help?

- **Streamlit Community:** [discuss.streamlit.io](https://discuss.streamlit.io)
- **Hugging Face Docs:** [huggingface.co/docs](https://huggingface.co/docs)
- **Railway Support:** [railway.app/docs](https://railway.app/docs)

Happy sharing! 🚀 