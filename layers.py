import tensorflow as tf
from utils import get_layer_uid


class Layer(object):
    def __init__(self, nonlin=tf.identity, N=1, name=None, logging=False):
        self.N = N
        if name is None:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.logging = logging
        self.nonlinearity = nonlin
        self.build()
        print 'Logging: {}'.format(self.logging)

    def __call__(self, x, sample=True, **kwargs):
        with tf.name_scope(self.name):
            if self.logging:
                tf.summary.histogram(self.name + '/inputs', x)
            output = self.call(x, sample=sample, **kwargs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', output)
            outputs = self.nonlinearity(output)
            return outputs

    def call(self, x, sample=True, **kwargs):
        raise NotImplementedError()

    def build(self):
        raise NotImplementedError()

    def f(self, x, sampling=True, **kwargs):
        raise NotImplementedError()

    def get_reg(self):
        return - (1. / self.N) * self.kldiv()

    def kldiv(self):
        raise NotImplementedError

from dense_layers import *
from conv_layers import *
