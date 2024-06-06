import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Load the dataset
df = pd.read_csv('tweets.csv')
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['tweet_text'], df['cyberbullying_type'], random_state=0)
# Preprocess the training set
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    # Tokenize the text
    words = word_tokenize(text.lower())

    # Remove stop words
    words = [word for word in words if word not in stop_words]

    # Stem the words
    words = [stemmer.stem(word) for word in words]

    # Join the words back into a string
    return ' '.join(words)

X_train = X_train.apply(preprocess)
X_test = X_test.apply(preprocess)

# Extract features from the data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the machine learning algorithm
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Convert the predicted labels to "Cyberbullying" or "Not Cyberbullying"
#y_pred = ["Cyberbullying" if label == "Cyberbullying" else "Not Cyberbullying" for label in y_pred]

# Evaluate the performance of the system
print('Accuracy:', accuracy_score(y_test, y_pred))

# Input new phrase
new_phrase = "hello"

# Preprocess and extract features from the new phrase
new_phrase = preprocess(new_phrase)
new_phrase_features = vectorizer.transform([new_phrase])

# Predict the label of the new phrase
new_phrase_pred = clf.predict(new_phrase_features)

# Convert the predicted label to "Cyberbullying" or "Not Cyberbullying"
new_phrase_pred = "not_cyberbullying" if new_phrase_pred[0] == "not_cyberbullying" else "cyberbullying"

print(new_phrase_pred)