import pandas as pd
import numpy as np

dataset=pd.read_csv('Womens Clothing E-Commerce Reviews.csv')

import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
corpus=[]
stemmer=PorterStemmer()
dataset['Review Text']=dataset['Review Text'].fillna('')
for i in range(0,len(dataset['Review Text'])):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review Text'][i])
    review=review.lower()
    review=review.split()
    review=[stemmer.stem(word) for word in review if word not in set(stopwords.words("english"))]
    review=' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
x=vectorizer.fit_transform(corpus).toarray()

dataset['Sentiment']=dataset['Rating']>=4

y=pd.get_dummies(dataset['Sentiment'])


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)

print(model.score(x_test,y_test))
