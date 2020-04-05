import numpy as np
import pandas as pd

dataset=pd.read_csv('Restaurant_Reviews.tsv',sep='\t')

import re 
import nltk
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer

stemmer=PorterStemmer()
corpus=[]

for i in range(0,len(dataset)):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    review=[stemmer.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
count=CountVectorizer()
x=count.fit_transform(corpus).toarray()
y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=10)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)
print(model.score(x_test,y_test))