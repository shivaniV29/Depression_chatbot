import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load the depression chatbot dataset (replace 'depression_dataset.csv' with your file)
df = pd.read_csv('/content/depression_dataset.csv',encoding='utf-8')

# Download NLTK resources (only required if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
# Data preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    cleaned_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(cleaned_tokens)

df['cleaned_title'] = df['title'].apply(preprocess_text)

df['category'] = df['response'].apply(lambda x: "Depression" if "depression" in x.lower() else "Non-Depression")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_title'], df['category'], test_size=0.2, random_state=42)


# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train an SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_tfidf)
# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
