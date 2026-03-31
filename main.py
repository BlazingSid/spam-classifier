import pandas as pd
import re
import string
import pickle

# load dataset
df = pd.read_csv("spam.tsv", sep="\t", names=["label", "text"])

# convert labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

# vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df["text"])
y = df["label"]

# split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# accuracy
print("Accuracy:", model.score(X_test, y_test))

# save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

