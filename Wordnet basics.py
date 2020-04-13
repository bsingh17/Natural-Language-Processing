from nltk.corpus import wordnet

syns=wordnet.synsets('good')

#printing the full set
print(syns)

#printing the single word
print(syns[0].lemmas()[0].name())

#printing the definition
print(syns[0].definition())

#printing the examples
print(syns[0].examples())
