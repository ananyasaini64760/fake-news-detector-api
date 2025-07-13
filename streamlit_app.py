import streamlit as st
import requests

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")

news_text = st.text_area("Paste a news article here:")

if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:
        with st.spinner("Checking..."):
            response = requests.post("http://127.0.0.1:8000/predict", json={"text": news_text})
            if response.status_code == 200:
                result = response.json()
                st.success(f"🧠 Prediction: **{result['label']}**")
                st.write(f"📊 Confidence: `{result['confidence'] * 100:.2f}%`")
            else:
                st.error("⚠️ Error connecting to FastAPI backend.")
