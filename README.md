# 🚀 SpamGuard AI (Spam Classifier Web App)

An AI-powered web application that detects whether a message is **Spam or Not Spam** in real-time using Natural Language Processing.

---

## ⚡ Features

* 🧠 Machine Learning model (Logistic Regression)
* 🔤 Text processing using TF-IDF
* 📊 Confidence score for predictions
* 🧾 Recent prediction history
* 🌐 Deployed live using Streamlit

---

## 🛠️ Tech Stack

* **Language:** Python
* **ML/NLP:** scikit-learn, TF-IDF
* **Frontend/UI:** Streamlit
* **Deployment:** Streamlit Cloud

---

## 📸 Screenshots

<img width="1919" height="891" alt="image" src="https://github.com/user-attachments/assets/537cd1d3-0821-4f23-b677-a49acae1873f" />


---

## 🌐 Live Demo

👉 https://spam-classifier-by-shahid.streamlit.app/

---

## 🧠 How It Works

1. Input message is cleaned (lowercase, remove punctuation)
2. Text is converted into numerical features using TF-IDF
3. Logistic Regression model predicts the result
4. App displays:

   * Spam / Not Spam
   * Confidence score

---

## 📂 Project Structure

```id="p7kq8d"
spam-classifier/
│
├── app.py
├── main.py
├── predict.py
├── model.pkl
├── vectorizer.pkl
├── spam.tsv
└── requirements.txt
```

---

## ⚙️ Installation (Run Locally)

```bash id="o6z6bz"
git clone https://github.com/BlazingSid/spam-classifier.git
cd spam-classifier
pip install -r requirements.txt
python main.py
streamlit run app.py
```

---

## 🎯 Future Improvements

* 🌍 Multi-language spam detection (Hinglish support)
* 📈 Improve model accuracy with more data
* 🧠 Deep learning model (LSTM / Transformers)
* 💾 Store user history in database
* 📱 Mobile optimized UI

---

## 👨‍💻 Author

**Shahid** 🚀

* GitHub: https://github.com/BlazingSId

---

This project is open-source 
