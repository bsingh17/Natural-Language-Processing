import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SKlearnClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB

documents=[(list(movie_reviews.words(fileid)),category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words=[]

for w in movie_reviews.words():
    all_words.append(w)
    
all_words=nltk.FreqDist(all_words)
print(all_words.most_common(15))
word_features=list(all_words.keys())[:3000]

def find_features(document):
    words=set(document)
    features={}
    for w in word_features:
        features[w]=(w in words)
    return features

#print(find_features(movie_reviews.words('neg/cv000_29416.txt')))
featuresets=[(find_features(rev),category) for (rev,category) in documents]

train=featuresets[:1900]
test=featuresets[1900:]

classifier=nltk.DecisionTreeClassifier.train(train)

print('accuracy',nltk.classify.accuracy(classifier,test))
