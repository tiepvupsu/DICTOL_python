from __future__ import print_function
import numpy as np


class BaseModel(object):
    """
    base dictionary learning model for classification
    """
    # def __init__(self)
    def predict(self, data):
        raise NotImplementedError


    def evaluate(self, data, label):
        pred = self.predict(data)
        acc = np.sum(pred == label)/float(len(label))
        print('accuracy = {:.2f} %'.format(100 * acc))
        return acc

