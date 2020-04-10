from __future__ import print_function
from . import utils
from .optimize import Lasso
from . import base
import numpy as np


class SRC(base.BaseModel):
    def __init__(self, lamb=0.01):
        self.lamb = lamb
        self.D = None
        self.train_range = None
        self.C = None
        self.num_classes = None

    def fit(self, Y_train, label_train):
        self.D = Y_train
        self.train_range = utils.label_to_range(label_train)
        self.num_classes = len(self.train_range) - 1

    def predict(self, Y, iterations=100):
        lasso = Lasso(self.D, self.lamb)
        lasso.fit(Y, iterations=iterations)
        X = lasso.coef_
        E = np.zeros((self.num_classes, Y.shape[1]))
        for i in range(self.num_classes):
            Xi = utils.get_block_row(X, i, self.train_range)
            Di = utils.get_block_col(self.D, i, self.train_range)
            R = Y - np.dot(Di, Xi)
            E[i, :] = (R*R).sum(axis=0)
        return utils.vec(np.argmin(E, axis=0) + 1)


def mini_test_unit():
    print('\n================================================================')
    print('Mini Unit test: Sparse Representation-based Classification (SRC)')
    dataset = 'myYaleB'
    N_train = 2
    Y_train, Y_test, label_train, label_test = utils.train_test_split(dataset, N_train)
    src = SRC(lamb=0.01)
    src.fit(Y_train, label_train)
    src.evaluate(Y_test, label_test)


def test_unit():
    print('\n================================================================')
    print('Unit test: Sparse Representation-based Classification (SRC)')
    dataset = 'myYaleB'
    N_train = 15
    Y_train, Y_test, label_train, label_test = utils.train_test_split(dataset, N_train)
    src = SRC(lamb=0.01)
    src.fit(Y_train, label_train)
    src.evaluate(Y_test, label_test)


if __name__ == '__main__':
    mini_test_unit()
