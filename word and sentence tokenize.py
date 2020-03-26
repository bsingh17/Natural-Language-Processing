from nltk import sent_tokenize,word_tokenize

example_sentences="Hello how are you doing? Is everything ok . Are you liking to work with the nltk library?"
print(sent_tokenize(example_sentences))
print(word_tokenize(example_sentences))