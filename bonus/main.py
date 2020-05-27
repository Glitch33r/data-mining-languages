import nltk

# nltk.download()  # for first we need to run this code

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

sentence = "studies studying cries children knives boys thousands offers routes volcanoes beaches dresses parties churches wishes"
punctuations = "?:!.,;-'s"
sentence_words = nltk.word_tokenize(sentence)
for word in sentence_words:
    if word in punctuations:
        sentence_words.remove(word)

sentence_words
print("{0:20}{1:20}".format("Word", "Lemma"))
for word in sentence_words:
    print("{0:20}{1:20}".format(word, wordnet_lemmatizer.lemmatize(word)))


print('\nExample with POS-tag')
print(wordnet_lemmatizer.lemmatize('better', 'a'))
print(wordnet_lemmatizer.lemmatize('feet', 'n'))
print(wordnet_lemmatizer.lemmatize('stripes', 'n'))
print(wordnet_lemmatizer.lemmatize('ate', 'v'))
