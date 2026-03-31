import streamlit as st
import pickle
import re
import string
import numpy as np

st.set_page_config(page_title="SpamGuard AI", page_icon="🚀", layout="wide")

# 🔥 ADVANCED UI
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #1a1a2e, #0f0f1a);
    color: white;
}

.block-container {
    padding: 30px;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 25px rgba(0,0,0,0.3);
}

.title {
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    opacity: 0.7;
    margin-bottom: 20px;
}

.result-box {
    font-size: 22px;
    font-weight: bold;
    padding: 15px;
    border-radius: 12px;
    margin-top: 15px;
}

.footer {
    text-align:center;
    opacity:0.6;
    margin-top:30px;
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

# layout
col1, col2 = st.columns([2,1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="title">🚀 SpamGuard AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Detect spam instantly using AI</div>', unsafe_allow_html=True)

    msg = st.text_area("Enter message", height=150)

    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("Analyze Message ⚡"):
        if msg.strip() == "":
            st.warning("Enter a message")
        else:
            cleaned = clean_text(msg)
            vec = vectorizer.transform([cleaned])

            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]
            confidence = np.max(prob) * 100

            result = "🚨 SPAM" if pred == 1 else "✅ NOT SPAM"

            # result UI
            if pred == 1:
                st.markdown(f'<div class="result-box" style="background:#ff4b4b;">{result} ({confidence:.2f}%)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box" style="background:#00c853;">{result} ({confidence:.2f}%)</div>', unsafe_allow_html=True)

            st.progress(int(confidence))

            # insight
            st.markdown("### 🧠 Insight")
            if pred == 1:
                st.write("This message contains patterns commonly found in spam (offers, urgency, rewards).")
            else:
                st.write("This message looks like normal human conversation.")

            # save history
            st.session_state.history.append((msg, result, confidence))

    st.markdown('</div>', unsafe_allow_html=True)

# 📊 SIDE PANEL
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 📊 Recent Checks")

    if st.session_state.history:
        for m, r, c in reversed(st.session_state.history[-5:]):
            st.write(f"**{r}**")
            st.write(f"{m[:40]}...")
            st.write(f"{c:.1f}%")
            st.markdown("---")
    else:
        st.write("No history yet")

    st.markdown("### 🌍 Language Support")
    st.write("Works best with English & Hinglish")

    st.markdown('</div>', unsafe_allow_html=True)

# footer
st.markdown('<div class="footer">Built by Shahid 🚀</div>', unsafe_allow_html=True)

