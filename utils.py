import numpy as np
import tensorflow as tf
import os, sys

_LAYER_UIDS = {}

tf.set_random_seed(1)
prng = np.random.RandomState(1)

flags = tf.app.flags
FLAGS = flags.FLAGS
sigma_init = 0.01

DATA_DIR = os.environ['DATA_DIR']
dtype = tf.float32


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def change_random_seed(seed):
    global prng
    prng = np.random.RandomState(seed)
    tf.set_random_seed(seed)


def randmat(shape, name, mu=0., type_init='he2', type_dist='normal', trainable=True, extra_scale=1.):
    if len(shape) == 1:
        dim_in, dim_out = shape[0], 0
    elif len(shape) == 2:
        dim_in, dim_out = shape
    else:
        dim_in, dim_out = np.prod(shape[1:]), shape[0]
    if type_init == 'xavier':
        bound = np.sqrt(1. / dim_in)
    elif type_init == 'xavier2':
        bound = np.sqrt(2. / (dim_in + dim_out))
    elif type_init == 'he':
        bound = np.sqrt(2. / dim_in)
    elif type_init == 'he2':
        bound = np.sqrt(4. / (dim_in + dim_out))
    elif type_init == 'regular':
        bound = sigma_init
    else:
        raise Exception()
    if type_dist == 'normal':
        val = tf.random_normal(shape, mean=mu, stddev=extra_scale * bound, dtype=dtype)  # actual weight initialization
    else:
        val = tf.random_uniform(shape, minval=mu - extra_scale * bound, maxval=mu + extra_scale * bound, dtype=dtype)

    return tf.Variable(initial_value=val, name=name, trainable=trainable)


def ones_d(shape):
    if isinstance(shape, (list, tuple)):
        shape = tf.stack(shape)
    return tf.ones(shape)


def zeros_d(shape):
    if isinstance(shape, (list, tuple)):
        shape = tf.stack(shape)
    return tf.zeros(shape)


def random_bernoulli(shape, p=0.5):
    if isinstance(shape, (list, tuple)):
        shape = tf.stack(shape)
    return tf.where(tf.random_uniform(shape) < p, tf.ones(shape), tf.zeros(shape))


def outer(x, y):
    return tf.matmul(tf.expand_dims(x, 1), tf.transpose(tf.expand_dims(y, 1)))

