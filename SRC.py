from __future__ import print_function
import utils
from sparse_coding import Lasso
import numpy as np


class SRC(object):
    def __init__(self, lamb = 0.01):
        self.lamb = lamb

    def fit(self, Y_train, label_train):
        self.D = Y_train
        self.train_range = utils.label_to_range(label_train)
        self.C = self.train_range.size - 1

    def predict(self, Y, verbose = True, iterations = 100):
        X = Lasso(self.D, self.lamb).solve(Y, iterations = 100)
        E = np.zeros((self.C, Y.shape[1]))
        for i in range(self.C):
            Xi = utils.get_block_row(X, i, self.train_range)
            Di = utils.get_block_col(self.D, i, self.train_range)
            R = Y - np.dot(Di, Xi)
            E[i,:] = (R*R).sum(axis = 0)
        return utils.vec(np.argmin(E, axis = 0) + 1)

    def evaluate(self, Y_test, label_test):
       pred = self.predict(Y_test)
       acc = np.sum(pred == label_test)/float(utils.numel(label_test))
       print('accuracy = %.2f'%(100*acc))
       return acc


def test_src():
    dataset = 'myYaleB'
    N_train = 10
    dataset, Y_train, Y_test, label_train, label_test = \
           utils.train_test_split(dataset, N_train)
    clf = SRC(lamb = 0.001)
    clf.fit(Y_train, label_train)
    clf.evaluate(Y_test, label_test)

if __name__ == '__main__':
    test_src()
