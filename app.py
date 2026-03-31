import streamlit as st
import pickle
import re
import string

st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

# 🔥 UI + particles
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f0f1a, #1a1a2e);
    color: white;
}

.block-container {
    background: rgba(255, 255, 255, 0.05);
    padding: 40px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    position: relative;
    z-index: 1;
}

.stTextInput>div>div>input {
    border-radius: 12px;
    padding: 12px;
    background: rgba(255,255,255,0.1);
    color: white;
}

.stButton>button {
    border-radius: 12px;
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    color: white;
    font-weight: bold;
    transition: 0.3s;
    padding: 10px 20px;
}

.stButton>button:hover {
    transform: scale(1.08);
    box-shadow: 0px 0px 20px rgba(100,100,255,0.6);
}

#bg {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
}
</style>

<canvas id="bg"></canvas>

<script>
const canvas = document.getElementById("bg");
const ctx = canvas.getContext("2d");

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

let particles = [];

for (let i = 0; i < 80; i++) {
  particles.push({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    vx: (Math.random() - 0.5) * 1.5,
    vy: (Math.random() - 0.5) * 1.5
  });
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let p of particles) {
    p.x += p.vx;
    p.y += p.vy;

    if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
    if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

    ctx.beginPath();
    ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
    ctx.fillStyle = "white";
    ctx.fill();
  }

  requestAnimationFrame(draw);
}

draw();
</script>
""", unsafe_allow_html=True)

# load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

# UI
st.markdown("<h1 style='text-align:center;'>📩 Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered message checker</p>", unsafe_allow_html=True)

msg = st.text_input("Enter your message")

if st.button("🚀 Predict"):
    if msg.strip() == "":
        st.warning("Enter a message first")
    else:
        cleaned = clean_text(msg)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)

        if pred[0] == 1:
            st.error("🚨 This is SPAM")
        else:
            st.success("✅ This is NOT spam")

# force redeploy