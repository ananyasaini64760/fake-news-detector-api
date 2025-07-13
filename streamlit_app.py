import streamlit as st
import requests

# Set your FastAPI backend URL here
API_URL = "http://127.0.0.1:8000/predict"  #  Change to your Render URL when deployed

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("Enter a news article and find out if it's **Real** or **Fake** using our AI model.")

# Text input
news_text = st.text_area("Paste your news text here:", height=200)
response = requests.post(API_URL, json={"text": news_text})
# Prediction
if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(API_URL, json={"text": news_text})
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"🧠 **Prediction:** {result['label']}")
                    st.info(f"🔍 **Confidence:** {result['confidence'] * 100:.2f}%")
                else:
                    st.error("❌ Failed to get a valid response from the API.")
            except Exception as e:
                st.error(f"❌ Error contacting API: {e}")
