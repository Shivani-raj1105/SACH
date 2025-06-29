try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ModuleNotFoundError:
    print("[ERROR] Streamlit is not installed. Please install it using 'pip install streamlit'")
    STREAMLIT_AVAILABLE = False

import requests
import csv
import os
import pandas as pd

FEEDBACK_FILE = "feedback.csv"

if STREAMLIT_AVAILABLE:
    # Custom CSS for luxury look
    st.markdown("""
    <style>
    body, .stApp {
        background-color: #181818;
        color: #e5c07b;
        font-family: 'Georgia', serif;
    }
    .sidebar .sidebar-content {
        background: #222;
        color: #e5c07b;
    }
    .block-container {
        background: #222;
        border-radius: 18px;
        box-shadow: 0 4px 32px 0 rgba(0,0,0,0.25);
        padding: 2rem 2rem 2rem 2rem;
    }
    .lux-title {
        font-size: 2.8rem;
        font-family: 'Georgia', serif;
        color: #FFD700;
        letter-spacing: 2px;
        text-shadow: 0 2px 8px #000, 0 0px 2px #FFD700;
        margin-bottom: 0.5em;
    }
    .lux-card {
        background: #181818;
        border: 1.5px solid #FFD700;
        border-radius: 16px;
        padding: 1.5rem 1.5rem 1.2rem 1.5rem;
        box-shadow: 0 2px 16px 0 rgba(255,215,0,0.08);
        margin-bottom: 1.5rem;
    }
    .lux-label {
        font-size: 2.2rem;
        font-weight: bold;
        color: #FFD700;
        text-shadow: 0 1px 4px #000;
    }
    .lux-fake {
        color: #ff4c4c;
        font-size: 2.2rem;
        font-weight: bold;
        text-shadow: 0 1px 4px #000;
    }
    .lux-real {
        color: #50fa7b;
        font-size: 2.2rem;
        font-weight: bold;
        text-shadow: 0 1px 4px #000;
    }
    .lux-progress .stProgress > div > div > div {
        background-image: linear-gradient(90deg, #FFD700, #e5c07b);
    }
    .lux-footer {
        color: #FFD700;
        text-align: center;
        font-size: 1.1rem;
        margin-top: 2rem;
        letter-spacing: 1px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar with luxury style
    st.sidebar.markdown("""
    <div style='color:#FFD700; font-family:Georgia,serif; font-size:1.5rem; text-align:center; margin-bottom:1.5rem;'>
        <b>📰 Luxe News Detector</b>
    </div>
    <hr style='border:1px solid #FFD700;'>
    <div style='color:#e5c07b; font-size:1.1rem;'>
    <b>Instructions:</b><br>
    - Paste a news headline or article.<br>
    - Click <b>Check if it's Fake</b>.<br>
    - See the luxurious result below.<br><br>
    <b>About:</b><br>
    This app uses a BERT-based model to detect fake news with style.<br>
    </div>
    <hr style='border:1px solid #FFD700;'>
    <div style='color:#FFD700; font-size:1rem;'>
    <b>Created by:</b> Your Name<br>
    <b>Contact:</b> your.email@example.com
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio("Navigation", ["Detector", "Visualizations"])

    if page == "Detector":
        st.markdown('<div class="lux-title">🧠 Luxe Fake News Detector</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 3])

        with col1:
            st.markdown('<div class="lux-card">', unsafe_allow_html=True)
            st.subheader("Enter News Text", divider="orange")
            input_text = st.text_area("Paste news headline or article:", height=200)
            analyze = st.button("Check if it's Fake")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="lux-card">', unsafe_allow_html=True)
            st.subheader("Prediction Result", divider="orange")
            feedback_given = False
            if analyze and input_text.strip():
                with st.spinner("Analyzing..."):
                    try:
                        res = requests.post("http://127.0.0.1:8000/predict", json={"text": input_text})
                        result = res.json()
                        label = result.get('label', 'Unknown')
                        confidence = result.get('confidence', 0)
                        explanation = result.get('explanation', [])

                        # Color-coded, luxury result
                        if label == "Fake":
                            st.markdown('<div class="lux-fake">🛑 FAKE</div>', unsafe_allow_html=True)
                        elif label == "Real":
                            st.markdown('<div class="lux-real">✅ REAL</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="lux-label">{label}</div>', unsafe_allow_html=True)

                        # Confidence progress bar (luxury style)
                        st.markdown('<div class="lux-progress">', unsafe_allow_html=True)
                        st.progress(min(max(confidence, 0), 1))
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.write(f"Confidence: <b>{confidence:.2%}</b>", unsafe_allow_html=True)

                        # LIME explanation visualization (fixed)
                        if explanation and len(explanation) > 0:
                            st.markdown('<hr style="border:1px solid #FFD700;">', unsafe_allow_html=True)
                            st.markdown('<b>Top Influential Words:</b>', unsafe_allow_html=True)
                            
                            # Create proper dataframe for visualization
                            try:
                                words, weights = zip(*explanation)
                                df_explanation = pd.DataFrame({
                                    'Word': words,
                                    'Importance': weights
                                })
                                st.bar_chart(df_explanation.set_index('Word'))
                                
                                # Highlight words in the input text
                                highlighted = input_text
                                for word, weight in explanation:
                                    if word.strip() and word in highlighted:
                                        color = "#FFD700" if weight > 0 else "#ff4c4c"
                                        highlighted = highlighted.replace(word, f'<span style="background-color:{color};border-radius:4px;">{word}</span>')
                                st.markdown(f'<div style="line-height:1.8;font-size:1.1rem;">{highlighted}</div>', unsafe_allow_html=True)
                            except Exception as e:
                                st.info("Word analysis available but visualization could not be displayed.")
                                st.write("**Key words:**", ", ".join([word for word, _ in explanation[:5]]))

                        # User feedback button
                        if st.button("Flag as Incorrect Prediction"):
                            # Save feedback to CSV
                            feedback_data = [input_text, label, confidence, str(explanation)]
                            file_exists = os.path.isfile(FEEDBACK_FILE)
                            with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as f:
                                writer = csv.writer(f)
                                if not file_exists:
                                    writer.writerow(["text", "model_prediction", "confidence", "explanation"])
                                writer.writerow(feedback_data)
                            st.success("Thank you for your feedback! This prediction has been flagged for review.")
                            feedback_given = True
                    except Exception as e:
                        st.error(f"API Error: {e}")
            elif analyze:
                st.warning("Please enter some text to analyze.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Footer
        st.markdown("""
        <div class='lux-footer'>
            <hr style='border:1px solid #FFD700;'>
            <small>© 2024 <b>Your Name</b>. All rights reserved. | Luxe Edition</small>
        </div>
        """, unsafe_allow_html=True)
    elif page == "Visualizations":
        st.markdown('<div class="lux-title">📊 Visualizations & Insights</div>', unsafe_allow_html=True)
        st.markdown('<div class="lux-card">', unsafe_allow_html=True)
        st.subheader("Feedback Data Insights", divider="orange")
        if os.path.isfile(FEEDBACK_FILE):
            df = pd.read_csv(FEEDBACK_FILE)
            # Class distribution
            st.markdown("**Flagged Prediction Distribution:**")
            st.bar_chart(df['model_prediction'].value_counts())
            # Show a sample of flagged feedback
            st.markdown("**Sample Flagged Feedback:**")
            st.dataframe(df.sample(min(5, len(df))))
        else:
            st.info("No feedback data available yet.")
        st.markdown('</div>', unsafe_allow_html=True)
        # Optionally, add more metrics or confusion matrix if you have ground truth labels in feedback.csv
