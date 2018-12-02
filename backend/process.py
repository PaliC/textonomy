import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import numpy as np
from scipy import sparse
import re
from gram_maker import make_n_grams
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


def make_vectors(x, y):
    word_to_index = {}

    words = set()
    for line in x:
        for word in line:
            words.add(word)
    root_to_word = stem_group(list(words))

    # count of distinct roots for indexing
    count = 0
    for root in root_to_word:
        for word in root_to_word[root]:
            word_to_index[word] = count
        count += 1

    all_vec = []

    print(len(word_to_index), " words")

    line_num = 0
    for line in x:
        indexes = list(map(lambda x: word_to_index[x], line))
        vec = make_binary(indexes, len(word_to_index))
        all_vec.append(vec)
        line_num += 1
        # print(line_num/len(x)*100, "% complete")

    X = np.matrix(np.stack(all_vec))
    Y = np.matrix(y).T

    X = sparse.csr_matrix(X)

    return X, Y, word_to_index


def make_test_vectors(x, y, word_to_index):
    all_vec = []
    line_num = 0
    for line in x:
        indexes = []
        for word in line:
            if word in word_to_index:
                indexes.append(word_to_index[word])
        vec = make_binary(indexes, len(word_to_index))
        all_vec.append(vec)
        line_num += 1
        # print(line_num/len(x)*100, "% complete")

    X = np.matrix(np.stack(all_vec))
    Y = np.matrix(y).T
    return X, Y


if __name__ == "__main__":
    xtrain, xtest, ytrain, ytest = process()
    print(make_vectors(xtrain, ytrain))

