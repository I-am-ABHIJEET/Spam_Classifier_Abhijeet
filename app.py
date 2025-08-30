import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# =======================
# NLTK Setup
# =======================
ps = PorterStemmer()

# =======================
# Preprocessing function
# =======================
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# =======================
# Load trained model
# =======================
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# =======================
# Page Config
# =======================
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

# =======================
# Custom CSS for beauty ✨
# =======================
st.markdown(
    """
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #d9a7c7, #fffcdc);
    }
    /* Title animation */
    .title {
        font-size: 40px !important;
        font-weight: 700;
        text-align: center;
        color: #2c3e50 !important;
        animation: fadeIn 2s ease-in-out;
        margin-bottom: 20px;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(-20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    /* Prediction result */
    .result {
        font-size: 28px;
        font-weight: 600;
        text-align: center;
        padding: 15px;
        border-radius: 15px;
        margin-top: 20px;
    }
    .spam {
        background-color: #ff6b6b;
        color: white;
        box-shadow: 0 4px 15px rgba(255, 0, 0, 0.4);
        animation: pulse 1.5s infinite;
    }
    .not-spam {
        background-color: #1dd1a1;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 200, 100, 0.4);
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% {transform: scale(1);}
        50% {transform: scale(1.05);}
        100% {transform: scale(1);}
    }
    /* Hidden signature */
    .signature {
        font-size: 10px;
        color: #999;
        text-align: right;
        opacity: 0.2;
        position: fixed;
        bottom: 5px;
        right: 10px;
    }
    /* Custom: Black label and warning */
    .black-label {
        color: #111 !important;
        font-weight: 600;
        font-size: 20px !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .big-emoji {
        font-size: 28px !important;
        margin-right: 6px;
        vertical-align: middle;
    }
    .black-warning {
        color: #111 !important;
        background: none !important;
        font-weight: 600;
        font-size: 18px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =======================
# UI
# =======================
st.markdown('<div class="title">📩 Email / SMS Spam Classifier</div>', unsafe_allow_html=True)

# Custom label aligned with textbox (removed bottom margin/padding)
st.markdown(
    """
    <div style="display: flex; align-items: center; margin: 0px 0px -5px 2px; line-height: 1;">
        <span style="font-size:20px; margin-right:4px; vertical-align:middle;">✍️</span>
        <span style="color:#111; font-weight:500; font-size:15px; vertical-align:middle;">Enter your message here:</span>
    </div>
    """,
    unsafe_allow_html=True
)
input_sms = st.text_area("", key="input_sms")

if st.button("🔍 Predict"):
    if input_sms.strip() == "":
        st.markdown('<div class="black-warning">⚠️ Please enter a message to classify.</div>', unsafe_allow_html=True)
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]

        # 4. Display with animation
        if result == 1:
            st.markdown('<div class="result spam">🚨 Spam Message</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result not-spam">✅ Not Spam</div>', unsafe_allow_html=True)

# Hidden signature
st.markdown('<div class="signature">Made by Abhijeet</div>', unsafe_allow_html=True)
