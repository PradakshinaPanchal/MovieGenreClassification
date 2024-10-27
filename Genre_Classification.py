import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os

# Ensure NLTK data is available, with error handling
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('stopwords')
    nltk.download('punkt')

# Helper function to clean text
def clean_text(text):
    """Removes non-alphabet characters and stopwords from the text."""
    stop_words = set(stopwords.words('english'))  # Load stopwords
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

# 1. Load Train Data
train_data = []
try:
    with open('train_data.txt', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ::: ')
            if len(parts) == 4:  # Ensure correct format
                train_data.append(parts)
except Exception as e:
    print(f"Error loading train.txt: {e}")

print(f"Train data loaded: {len(train_data)} rows")

# 2. Load Test Data
test_data = []
try:
    with open('test_data.txt', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ::: ')
            if len(parts) == 3:  # Ensure correct format
                test_data.append(parts)
except Exception as e:
    print(f"Error loading test.txt: {e}")

print(f"Test data loaded: {len(test_data)} rows")

# Convert train and test data to DataFrames
train_df = pd.DataFrame(train_data, columns=['ID', 'Title', 'Genre', 'Description'])
test_df = pd.DataFrame(test_data, columns=['ID', 'Title', 'Description'])

print("Train Data Sample:")
print(train_df.head())

print("Test Data Sample:")
print(test_df.head())

# 3. Clean Descriptions
train_df['Clean_Description'] = train_df['Description'].apply(clean_text)
test_df['Clean_Description'] = test_df['Description'].apply(clean_text)

print("Cleaned Descriptions:")
print(train_df[['Description', 'Clean_Description']].head())

# 4. Extract Features using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)

# Transform the descriptions into numerical features
X_train = tfidf.fit_transform(train_df['Clean_Description'])
y_train = train_df['Genre']  # Target labels
X_test = tfidf.transform(test_df['Clean_Description'])

print("TF-IDF Feature Extraction Completed.")

# 5. Train Logistic Regression Model
try:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("Model training completed.")
except Exception as e:
    print(f"Error during model training: {e}")

# 6. Predict Genres for Test Data
try:
    test_df['Predicted_Genre'] = model.predict(X_test)
    print("Prediction completed.")
except Exception as e:
    print(f"Error during prediction: {e}")

# 7. Output the Results
print("Test Data Predictions:")
print(test_df[['ID', 'Title', 'Predicted_Genre']])