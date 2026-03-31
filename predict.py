import pickle
import re
import string

# load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# clean text (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

print("=== Spam Classifier ===")

while True:
    msg = input("\nEnter message (or type 'exit'): ")

    if msg.lower() == "exit":
        print("Goodbye 👋")
        break

    if msg.strip() == "":
        print("⚠️ Enter something")
        continue

    cleaned = clean_text(msg)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)

    if pred[0] == 1:
        print("🚨 Spam")
    else:
        print("✅ Not Spam")
