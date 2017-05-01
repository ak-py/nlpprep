from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = "I want to deter competition from coming in."

stop_words = set(stopwords.words("english"))

#print(stop_words)

words = word_tokenize(example_sentence)

filtered_sentence = [w for w in words if not w in stop_words]

print(filtered_sentence)

example_sentence = "I want to drive the competition out."

words = word_tokenize(example_sentence)

filtered_sentence = [w for w in words if not w in stop_words]

print(filtered_sentence)
