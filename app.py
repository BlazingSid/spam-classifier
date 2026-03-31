import streamlit as st
import pickle
import re
import string
import numpy as np

st.set_page_config(page_title="Spam Classifier Pro", page_icon="🚀", layout="centered")

# 🔥 STYLING (clean + pro)
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f0f1a, #1a1a2e);
    color: white;
}

.block-container {
    background: rgba(255,255,255,0.05);
    padding: 40px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
}

h1 {
    text-align: center;
    font-size: 40px;
}

.tagline {
    text-align: center;
    opacity: 0.7;
    margin-bottom: 20px;
}

.footer {
    text-align: center;
    margin-top: 30px;
    opacity: 0.6;
}
</style>
""", unsafe_allow_html=True)

# load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# clean
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

# UI
st.markdown("<h1>📩 Spam Detector Pro</h1>", unsafe_allow_html=True)
st.markdown("<p class='tagline'>AI-powered message intelligence</p>", unsafe_allow_html=True)

msg = st.text_area("Enter your message", height=120)

# history
if "history" not in st.session_state:
    st.session_state.history = []

if st.button("🚀 Analyze"):
    if msg.strip() == "":
        st.warning("Enter a message first")
    else:
        cleaned = clean_text(msg)
        vec = vectorizer.transform([cleaned])

        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        confidence = np.max(prob) * 100

        result = "Spam" if pred == 1 else "Not Spam"

        # display result
        if pred == 1:
            st.error(f"🚨 SPAM ({confidence:.2f}% confidence)")
        else:
            st.success(f"✅ NOT SPAM ({confidence:.2f}% confidence)")

        # progress bar
        st.progress(int(confidence))

        # save history
        st.session_state.history.append((msg, result, confidence))

# 📊 HISTORY SECTION
if st.session_state.history:
    st.markdown("### 🧾 Prediction History")

    for i, (m, r, c) in enumerate(reversed(st.session_state.history[-5:])):
        st.write(f"**{i+1}.** {m}")
        st.write(f"→ {r} ({c:.2f}%)")
        st.markdown("---")

# footer
st.markdown("<div class='footer'>Built by Shahid 🚀</div>", unsafe_allow_html=True)

