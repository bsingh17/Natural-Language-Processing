import numpy as np
import pandas as pd

train=pd.read_csv('train.csv')
train=train.drop(['id','keyword','location'],axis='columns')
test=pd.read_csv('test.csv')
test=test.drop(['id','keyword','location'],axis='columns')


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer=PorterStemmer()
corpus=[]
corpus1=[]
for i in range(0,len(train)):
    review=re.sub('[^a-zA-Z]',' ',train['text'][i])
    review=review.lower()
    review=review.split()
    review=[stemmer.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
count=CountVectorizer(max_features=4000)
x=count.fit_transform(corpus).toarray()

for i in range(0,len(test)):
    review=re.sub('[^a-zA-Z]',' ',test['text'][i])
    review=review.lower()
    review=review.split()
    review=[stemmer.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus1.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
count=CountVectorizer(max_features=4000 )
x_test=count.fit_transform(corpus1).toarray()

y=train['target'].astype(str)

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x,y)
y_predict=model.predict(x_test)