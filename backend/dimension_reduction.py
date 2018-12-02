from process import make_vectors, process, make_test_vectors
import numpy as np
from sklearn import random_projection
from sklearn.metrics.pairwise import cosine_similarity
import functools
import time
from multiprocessing import Pool, cpu_count, Value


class Counter(object):
    def __init__(self):
        self.e = Value('i', 0)
        self.t = Value('i', 0)

    def inc_error(self):
        with self.e.get_lock():
            self.e.value += 1

    def inc_total(self):
        with self.t.get_lock():
            self.t.value += 1

    def print_error(self):
        print(self.e.value/self.t.value*100,"% error.")


def cosine_similarity_wrap(vec_to_compare):
    return functools.partial(cosine_similarity, vec_to_compare)


def csp_partial(X, Y, Xtest):
    return functools.partial(cosine_similarity_pool_map, X, Y, Xtest)


def cosine_similarity_pool_map(X, Y, Xtest, params):
    cos_sim = cosine_similarity_wrap([Xtest[params[0],:]])
    max_sim = 0
    max_sim_indexes = []

    for i, row in enumerate(X):
        sim = cos_sim([row])
        if sim[0,0] > max_sim:
            max_sim = sim[0,0]
            max_sim_indexes = [i]
        elif sim[0,0] == max_sim:
            max_sim_indexes.append(i)

    cats = set(list(map(lambda x: Y[x,0], max_sim_indexes)))
    
    global count
    count.inc_total()

    if params[1] not in cats:
        count.inc_error()
        #count.print_error()
        return False

    return True

def pooled_test(X, Y, Xtest, Ytest):
    csp = csp_partial(X,Y,Xtest)
    p = Pool(cpu_count())
    correct_list = p.map(csp, list(zip(list(range(Xtest.shape[0])), \
            map(lambda x: x[0],Ytest.tolist()))))
    return


def t():
    global start
    try:
        end = time.time()
        print("time elapsed: ", end - start)
        start = time.time()
    except NameError:
        start = time.time()
    return 

def test(X, Y, Xtest, Ytest):
    errors = 0

    for i1, current_row in enumerate(Xtest):
        cos_sim = cosine_similarity_wrap([current_row])
        
        max_sim = 0
        max_sim_indexes = []

        for i2, row in enumerate(X):
            sim = cos_sim([row])
            if sim[0,0] > max_sim:
                max_sim = sim[0,0]
                max_sim_indexes = [i2]
            elif sim[0,0] == max_sim:
                max_sim_indexes.append(i2)
        
        t()
        print((i1*X.shape[0]+i2)/(X.shape[0]*Xtest.shape[0])*100, "%")
        cats = set(list(map(lambda x: Y[x,0], max_sim_indexes)))
        if Ytest[i1,0] not in cats:
            errors += 1
            print("errors: ", errors/(i1+1)*100, "%")
    return 

from sklearn.decomposition import TruncatedSVD

def run_test(comp, iters):
    global count
    count = Counter()
    xtrain, xtest, ytrain, ytest = process()
    X, Y, word_to_index = make_vectors(xtrain, ytrain) 

    svd = TruncatedSVD(n_components=comp, n_iter=iters, random_state=0)

    Xred = svd.fit_transform(X)
    Xtest, Ytest = make_test_vectors(xtest,ytest, word_to_index) 
    Xtest_red = svd.transform(Xtest)
    
    pooled_test(Xred, Y, Xtest_red, Ytest)
    
    return count.e.value/count.t.value*100

if __name__ == "__main__":
    
    for comp in range(1000, 3000, 1000):
        for iters in range(5, 10, 5):
            print("n_components=%d, n_iter=%d" % (comp, iters))
            start = time.time()
            error = run_test(comp, iters)
            end = time.time()
            print("error: %d" % error)
            print("elapsed time: %d" % (end - start))
            print()



