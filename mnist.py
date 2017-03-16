import gzip
import cPickle as pkl
import numpy as np


class MNIST(object):
    def __int__(self):
        self.nb_classes = 10
        self.name = self.__class__.__name__.lower()

    def load_data(self):
        with gzip.open('mnist.pkl.gz', 'rb') as f:
            try:
                train_set, valid_set, test_set = pkl.load(f, encoding='latin1')
            except:
                train_set, valid_set, test_set = pkl.load(f)
        return [train_set[0], train_set[1]], [valid_set[0], valid_set[1]], [test_set[0], test_set[1]]

    def permutation_invariant(self, n=None):
        train, valid, test = self.load_data()
        return train, valid, test

    def images(self, n=None):
        train, valid, test = self.load_data()
        train[0] = np.reshape(train[0], (train[0].shape[0], 1, 28, 28))
        valid[0] = np.reshape(valid[0], (valid[0].shape[0], 1, 28, 28))
        test[0] = np.reshape(test[0], (test[0].shape[0], 1, 28, 28))
        return train, valid, test
