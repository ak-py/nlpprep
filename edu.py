from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords

train_set = ("I want to deter competition from coming in.",
             "I know I can run a lean efficient restaurant and keep my costs low.",
             "I want my initial customers to try it.  If the volume is good, I can raise the price later.",
             "I prefer Lo margin Hi volume. Customers are happy if they feel they are getting good food for their money.",
             "The location at which the restaurant is, is such that it won’t support hi price.",
             "The market this burger is targeted at will prefer low price.",
             "It is preferable that prices are increased than decreased. Decreasing prices mean the burger is not doing well. That is a bad signal to send.")

test_set = ("I want to have low competition",
"We can see the shining sun, the bright sun.")



count_vectorizer = CountVectorizer(stop_words=set(stopwords.words("english")))

count_vectorizer.fit_transform(train_set)

print ("Vocabulary:", count_vectorizer.vocabulary_)
# Vocabulary: {'blue': 0, 'sun': 1, 'bright': 2, 'sky': 3}

print(count_vectorizer.stop_words)

freq_term_matrix = count_vectorizer.transform(test_set)

print (freq_term_matrix.todense())

#[[0 1 1 1]
#[0 2 1 0]]

tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)

print ("IDF:", tfidf.idf_)

# IDF: [ 0.69314718 -0.40546511 -0.40546511  0.        ]

tf_idf_matrix = tfidf.transform(freq_term_matrix)
print (tf_idf_matrix.todense())

# [[ 0.         -0.70710678 -0.70710678  0.        ]
# [ 0.         -0.89442719 -0.4472136   0.        ]]