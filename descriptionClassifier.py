import _pickle as pickle
def classify(str):
    with open('svd.pkl', 'rb') as input:
        svd = pickle.load(input)
    return str + "is classified"