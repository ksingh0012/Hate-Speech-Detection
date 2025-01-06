import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)

# Load model
model_file = 'SVM_hate_speech_model.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Load TF-IDF vectorizer
vectorizer_file = 'tfidf_vectorizer_hate.pkl'
with open(vectorizer_file, 'rb') as file:
    vectorizer = pickle.load(file)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function for cleaning text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|[^a-zA-Z\s]', '', text)  # Remove mentions
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Remove stopwords and lemmatize
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    cleaned_tweet = clean_text(tweet)
    tweet_vector = vectorizer.transform([cleaned_tweet])  # Now this will work as the vectorizer is fitted
    prediction = model.predict(tweet_vector)
    label = 'Hate Speech' if prediction[0] == 'Hate_Speech' else ('Offensive Language' if prediction[0] == 'offensive_language' else 'Neither')
    return render_template('index.html', prediction=label, tweet=tweet)

if __name__ == '__main__':
    app.run(debug=True)
