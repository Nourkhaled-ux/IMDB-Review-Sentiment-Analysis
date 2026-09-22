import streamlit as st
import pickle
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 IMDb Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below and the model will predict "
    "whether the sentiment is **Positive** or **Negative**."
)

review = st.text_area(
    "Movie Review",
    height=200,
    placeholder="Type your review here..."
)

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")
    else:

        cleaned_review = preprocess(review)
        review_vector = vectorizer.transform([cleaned_review])
        prediction = model.predict(review_vector)
        sentiment = label_encoder.inverse_transform(prediction)[0]
        st.subheader("Prediction")
        if sentiment.lower() == "positive":
            st.success("Positive Review")
        else:
            st.error("Negative Review")

        
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(review_vector)
            confidence = probability.max() * 100
            st.write(f"**Confidence:** {confidence:.2f}%")







