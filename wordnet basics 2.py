from nltk.corpus import wordnet

#printing synonyms and antonyms
synonyms=[]
antonyms=[]

for syns in wordnet.synsets('good'):
    for lma in syns.lemmas():
        synonyms.append(lma.name())
        if lma.antonyms():
            antonyms.append(lma.antonyms()[0].name())
print(antonyms)

#printing similarity percentage between two words

w1=wordnet.synset('ship.n.01')
w2=wordnet.synset('boat.n.01')
w3=wordnet.synset('plant.n.01')
w4=wordnet.synset('water.n.01')

print(w1.wup_similarity(w2))
print(w1.wup_similarity(w3))
print(w1.wup_similarity(w4))
