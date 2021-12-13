from os import name
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from nltk.stem.snowball import SnowballStemmer
from collections import Counter 
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
import wordcloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sqlalchemy import create_engine

# importing the words that awont be used during the sentimental analysis
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")
newStopWords = ['u','hotel','room','location','staff','go','got','via','or','ur','us','in','i','let','the','to','is','amp','make','one','day','days','get']
stopwords.extend(newStopWords)


# read reviews from db in MySQL
engine = create_engine('mysql+mysqlconnector://root:root@localhost/hotels')
dbConnection = engine.connect()
df = pd.read_sql("CALL GetAmountOfRows(100000)", dbConnection);
dbConnection.close()

# remove special charachters
df['new_review'] = df.review.str.replace("[^a-zA-Z#]", " ")

# finding the most frequent words and making them lowercase
all_words = []
for line in list(df['new_review']):
    words = line.split()
    for word in words:
        all_words.append(word.lower())

a=Counter(all_words).most_common(10)
print(a)
df.head(50)

#tokenization
df['new_review'] = df['new_review'].apply(lambda x: x.split())
def process(text):
    # Check characters to see if they are have punctuation
    nopunc = set(char for char in list(text) if char not in string.punctuation)
    # Join the characters to form the string.
    nopunc = " ".join(nopunc)
    # remove any stopwords if present
    return [word for word in nopunc.lower().split() if word.lower() not in stopwords]
df['new_review'] = df['new_review'].apply(process)
df.head(50)

# function that creates word clouds
def generate_wordcloud(words,colour):
    wordcloud = WordCloud(
    background_color= colour,
    max_words=200,
    stopwords=stopwords
    ).generate_from_frequencies(words)
    plt.figure(figsize=(10,9))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# creating a word cloud based on all the reviews
words = []
for line in df['new_review']:
    words.extend(line)
generate_wordcloud(Counter(words),'white')

# creating a word cloud based on the negative reviews
neg_words = []
for line in df['new_review'][df['sentiment']==0]:
    neg_words.extend(line)
generate_wordcloud(Counter(neg_words),'black')

# creating a word cloud based on the positive reviews
pos_words = []
for line in df['new_review'][df['sentiment']==1]:
    pos_words.extend(line)
generate_wordcloud(Counter(pos_words),'white')

# changes the list/array into a string
def string (text):
    to_return=''
    for i in list(text):
        to_return += str(i) + ' '
    to_return = to_return[:-1]
    return to_return
    
df['new_review'] = df['new_review'].apply(string)

# classification time 😤
x_train, x_test, y_train, y_test =  train_test_split(df['new_review'], 
      df['sentiment'], test_size = 0.2, random_state = 42)

print("training set :",x_train.shape,y_train.shape)
print("testing set :",x_test.shape,y_test.shape)

# measuring the term frequency-inverse document frequency 
count_vect = CountVectorizer(stop_words='english')
transformer = TfidfTransformer(norm='l2',sublinear_tf=True)

x_train_counts = count_vect.fit_transform(x_train)
x_train_tfidf = transformer.fit_transform(x_train_counts)

x_test_counts = count_vect.transform(x_test)
x_test_tfidf = transformer.transform(x_test_counts)
df.head(15)

# # random forest classification
model = RandomForestClassifier(n_estimators=200)
model.fit(x_train_tfidf, y_train)

predictions = model.predict(x_test_tfidf)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))

# # Logistic regression classification 
logmodel = LogisticRegression(random_state=400 )
logmodel.fit(x_train_tfidf,y_train)

log_predictions = logmodel.predict(x_test_tfidf)

print(confusion_matrix(y_test,log_predictions))
print(classification_report(y_test,log_predictions))
print(accuracy_score(y_test, log_predictions))

# GradientBoosting regression classification 
alg= GradientBoostingRegressor(n_estimators= 550, learning_rate= 0.1, max_depth= 3)
alg.fit(x_train_tfidf,y_train)

alg_predictions = logmodel.predict(x_test_tfidf)

print(confusion_matrix(y_test,alg_predictions))
print(classification_report(y_test,alg_predictions))
print(accuracy_score(y_test, alg_predictions))
