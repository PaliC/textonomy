import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import numpy as np
from scipy import sparse
import re
from gram_maker import make_n_grams_listy_listy_list, get_n_grams_count_listy_list
from fuzzy_string_group import stem_group 

def tokenize(text):
    text = re.sub(r'[^A-Za-z]', r' ', text)
    arr = text.split()
    arr = [x for x in arr if len(x) > 1]
    return arr

def process():
    data = pd.read_csv('../search_strings.csv', encoding = "ISO-8859-1")
    data['item_title'] = data['item_title'].str.lower()

    items = data[['item_title']]
    categories = data[['category']]

    items = items.applymap(tokenize)

    clean_data = items.join(categories)

    shuffled_data = shuffle(clean_data, random_state=0)

    xtrain, xtest, ytrain, ytest = train_test_split(shuffled_data['item_title'], shuffled_data['category'], random_state=0)

    return (xtrain, xtest, ytrain, ytest)


def make_binary(indexes, length):
    vec = np.zeros((1,length))
    for i in indexes:
        vec[0,i] = 1/len(indexes)
    return vec


def make_vectors_ngram(x, y):
    word_to_index = {}
    words = set()
    
    for j in range(1,6):
        all_grams = make_n_grams_listy_listy_list(j, x.tolist())
        for i in range(x.shape[0]):
            for ng in all_grams[i]:
                ng_tup = tuple(ng)
                if ng_tup is not ():
                    words.add(ng_tup)

    count = 0
    for word in words:
        word_to_index[word] = count
        count += 1

    all_vec = []

    flatten = lambda x: [item for sublist in x for item in sublist]
    total = set()

    x = x.tolist()
    for i in range(len(x)):
        indexes = []
        for j in range(1,6):
            ngrams = make_n_grams_listy_listy_list(j, [x[i]])[0]
            ngrams = list(map(lambda x: tuple(x), ngrams))
            ngrams = list(filter(lambda x: x is not(), ngrams))
            indexes.append(list(map(lambda x: word_to_index[x], ngrams)))
        indexes = flatten(indexes)
        vec = make_binary(indexes, len(word_to_index))
        all_vec.append(vec)
        print(i/len(x)*100,"%")

    X = np.matrix(np.stack(all_vec))
    Y = np.matrix(y).T

    return X, Y, word_to_index


def make_test_vectors_ngram(x, y, word_to_index):
    all_vec = []

    flatten = lambda x: [item for sublist in x for item in sublist]
    total = set()

    x = x.tolist()
    for i in range(len(x)):
        indexes = []
        for j in range(1,6):
            ngrams = make_n_grams_listy_listy_list(j, [x[i]])[0]
            ngrams = list(map(lambda x: tuple(x), ngrams))
            ngrams = list(filter(lambda x: x is not() and x in word_to_index, ngrams))
            indexes.append(list(map(lambda x: word_to_index[x], ngrams)))
        indexes = flatten(indexes)
        vec = make_binary(indexes, len(word_to_index))
        all_vec.append(vec)
        print(i/len(x)*100,"%")

    X = np.matrix(np.stack(all_vec))
    Y = np.matrix(y).T
    return X, Y


if __name__ == "__main__":
    xtrain, xtest, ytrain, ytest = process()
    X,Y, word_to_index = make_vectors(xtrain, ytrain)
    Xtest, Ytest = make_test_vectors(xtest, ytest, word_to_index)
    
    
