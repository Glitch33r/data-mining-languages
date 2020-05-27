from pprint import pprint

sentence = """Following mice attacks, caring farmers were marching to Delhi for better living conditions. 
Delhi police on Tuesday fired water cannons and teargas shells at protesting farmers as they tried to 
break barricades with their cars, automobiles and tractors."""

# NLTK
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


pprint(
    " ".join(
        [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence) if
         w not in string.punctuation])
)

# ('Following mouse attack care farmer be march to Delhi for well living '
#  'condition Delhi police on Tuesday fire water cannon and teargas shell at '
#  'protest farmer a they try to break barricade with their car automobile and '
#  'tractor')

# Spacy
import spacy

nlp = spacy.load('en', disable=['parser', 'ner'])
doc = nlp(sentence)
pprint(" ".join([token.lemma_ for token in doc]))

# ('follow mice attack , care farmer be march to delhi for good living condition '
#  '. delhi police on tuesday fire water cannon and teargas shell at protest '
#  'farmer as -PRON- try to break barricade with -PRON- car , automobile and '
#  'tractor .')

# TextBlob
from textblob import TextBlob, Word


def lemmatize_with_postag(sentence):
    sent = TextBlob(sentence)
    tag_dict = {"J": 'a',
                "N": 'n',
                "V": 'v',
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return " ".join(lemmatized_list)


pprint(lemmatize_with_postag(sentence))

# ('Following mouse attack care farmer be march to Delhi for good living '
#  'condition Delhi police on Tuesday fire water cannon and teargas shell at '
#  'protest farmer a they try to break barricade with their car automobile and '
#  'tractor')

# Pattern
from pattern.en import lemma

pprint(" ".join([lemma(wd) for wd in sentence.split()]))

# ('follow mice attacks, care farmer be march to delhi for better live '
#  'conditions. delhi police on tuesday fire water cannon and tearga shell at '
#  'protest farmer a they try to break barricade with their cars, automobile and '
#  'tractors.')

# Stanford
from stanfordcorenlp import StanfordCoreNLP
import json, string


def lemmatize_corenlp(conn_nlp, sentence):
    props = {
        'annotators': 'pos,lemma',
        'pipelineLanguage': 'en',
        'outputFormat': 'json'
    }

    # tokenize into words
    sents = conn_nlp.word_tokenize(sentence)

    # remove punctuations from tokenised list
    sents_no_punct = [s for s in sents if s not in string.punctuation]

    # form sentence
    sentence2 = " ".join(sents_no_punct)

    # annotate to get lemma
    parsed_str = conn_nlp.annotate(sentence2, properties=props)
    parsed_dict = json.loads(parsed_str)

    # extract the lemma for each word
    lemma_list = [v for d in parsed_dict['sentences'][0]['tokens'] for k, v in d.items() if k == 'lemma']

    # form sentence and return it
    return " ".join(lemma_list)


nlp = StanfordCoreNLP('http://localhost', port=9000, timeout=30000)

pprint(lemmatize_corenlp(conn_nlp=nlp, sentence=sentence))

# ('follow mouse attack care farmer be march to Delhi for better living '
#  'condition Delhi police on Tuesday fire water cannon and tearga shell at '
#  'protest farmer as they try to break barricade with they car automobile and '
#  'tractor')
