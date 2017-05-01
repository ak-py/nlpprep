import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

#print (nltk.__file__)


documents = [(list(movie_reviews.words(fileid)),category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = {w in words}

    return features

#print((find_features(movie_reviews.words('neg/cv00_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]

testing_set = featuresets[1900:]

#classifier = nltk.NaiveBayesClassifier.train(training_set)

#print("Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier, training_set))*100)

#print(training_set)

#print(testing_set)

MNB_classifier= SklearnClassifier(MultinomialNB())
