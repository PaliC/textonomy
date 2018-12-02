from process import make_vectors, process, make_test_vectors
from lshash import LSHash


xtrain, xtest, ytrain, ytest = process()
X, Y, index_to_word, word_to_index = make_vectors(xtrain, ytrain)
Xtest, Ytest = make_test_vectors(xtest,ytest, word_to_index) 


hash_size = 100



def make_planes(hash_size, dim):
    np.random.randn(hash_size, dim)


