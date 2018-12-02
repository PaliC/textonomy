import sys

sys.path.append("./backend")

import _pickle as pickle
import functools
import re
from process import tokenize, make_binary
from dimension_reduction import cosine_similarity_wrap
import numpy as np


def classify_wrapper(svd_matrix, svd_object, word_to_index, Y):
    return functools.partial(classify_func, svd_matrix, svd_object, word_to_index, Y)


def classify_func(svd_matrix, svd_object, word_to_index, Y, string):
    words = tokenize(string)
    words = [i.lower() for i in words]
    indexes = []
    for word in words:
        if word in word_to_index:
            indexes.append(word_to_index[word])
    vec = make_binary(indexes, len(word_to_index))
    vec = svd_object.transform(vec)
    
    cos_sim = cosine_similarity_wrap(vec)
    max_sim = 0
    max_sim_indexes = []

    for i2, row in enumerate(svd_matrix):
        sim = cos_sim([row])
        if sim[0,0] > max_sim:
            max_sim = sim[0,0]
            max_sim_indexes = [i2]
        elif sim[0,0] == max_sim:
            max_sim_indexes.append(i2)

    cats = list(set(map(lambda x: Y[x,0], max_sim_indexes)))
    return (", ".join(cats))
    

with open('Y.pkl', 'rb') as input:
    Y = pickle.load(input)

with open('svd.pkl', 'rb') as input:
    svd = pickle.load(input)

with open('Xred.pkl', 'rb') as input:
    Xred = pickle.load(input)

with open('word_to_index.pkl', 'rb') as input:
    word_to_index = pickle.load(input)

classify = classify_wrapper(Xred, svd, word_to_index, Y)

# with open('classify.pkl', 'wb') as output:
#     pickle.dump(classify, output, -1)

