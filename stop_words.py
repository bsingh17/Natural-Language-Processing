from nltk.corpus import stopwords
from nltk import word_tokenize

example_sentence="Hello is it you there Amaan?"
words=word_tokenize(example_sentence)
stop_words=set(stopwords.words("english"))
filtered_sentence=[]

for i in words:
    if i not in stop_words:
        filtered_sentence.append(i)