import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class movie_modeling():
    X = pd.Series()
    y = pd.Series()
    y_test = []
    preds = []
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def removeStop(self, story):
        stop = set(stopwords.words("english"))
        copy = story[:]
        copy = " ".join([word.lower() for word in copy.split() if word.lower() not in stop])
        return copy
    
    def removePunc(self, story):
        copy = story[:]
        for char in story:
            if char in "`~!@#$%^&*()-_=+[]{}\\|;:'\",<.>/?":
                copy = copy.replace(char, "")
        return copy
    
    def lemmatize(self, story):
        lemmatizer = WordNetLemmatizer()
        copy = ""
        for word in story.split(" "):
            copy += lemmatizer.lemmatize(word) + " "
        return copy.strip(" ")
    
    def report(self):
        le = LabelEncoder()
        le.fit(list(set(self.y)))
        cr = classification_report(le.inverse_transform(self.preds), le.inverse_transform(self.y_test), output_dict = True)
        df = pd.DataFrame(cr)
        df.drop('support', inplace = True)
        df.drop(['accuracy'], axis = 1, inplace = True)
        return sns.heatmap(df.iloc[:-1, :].T, annot=True)
    
    def accuracy(self):
        return accuracy_score(self.preds, self.y_test)
    
    def model(self):
        le = LabelEncoder()
        le.fit(list(set(self.y)))
        X_cleaned = []
        for story in self.X:
            X_cleaned.append(self.lemmatize(self.removeStop(self.removePunc(story))))
        y_cleaned = le.transform(self.y)
        X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size = 0.25)
        self.y_test = y_test
        vectorizer = TfidfVectorizer()
        X_train_TFIDF = vectorizer.fit_transform(X_train)
        X_test_TFIDF = vectorizer.transform(X_test)
        model = LogisticRegression()
        model.fit(X_train_TFIDF, y_train)
        self.preds = model.predict(X_test_TFIDF)