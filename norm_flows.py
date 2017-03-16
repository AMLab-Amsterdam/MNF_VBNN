import tensorflow as tf
from utils import randmat, get_layer_uid, zeros_d, random_bernoulli


class MaskedNVPFlow(object):
    """
    """
    def __init__(self, incoming, n_flows=2, n_hidden=0, dim_h=10, name=None, scope=None, nonlin=tf.nn.tanh, **kwargs):
        self.incoming = incoming
        self.n_flows = n_flows
        self.n_hidden = n_hidden
        if name is None:
            layer = self.__class__.__name__.lower()
            self.name = layer + '_' + str(get_layer_uid(layer))
        else:
            self.name = name
        self.dim_h = dim_h
        self.params = []
        self.nonlin = nonlin
        self.scope = scope
        self.build()
        print 'MaskedNVP flow {} with length: {}, n_hidden: {}, dim_h: {}, name: {}, ' \
              'scope: {}'.format(self.name, n_flows, n_hidden, dim_h, name, scope)

    def build_mnn(self, fid, param_list):
        dimin = self.incoming
        with tf.variable_scope(self.scope):
            w = randmat((dimin, self.dim_h), name='w{}_{}_{}'.format(0, self.name, fid))
            b = tf.Variable(tf.zeros((self.dim_h,)), name='b{}_{}_{}'.format(0, self.name, fid))
            param_list.append([(w, b)])
            for l in xrange(self.n_hidden):
                wh = randmat((self.dim_h, self.dim_h), name='w{}_{}_{}'.format(l + 1, self.name, fid))
                bh = tf.Variable(tf.zeros((self.dim_h,)), name='b{}_{}_{}'.format(l + 1, self.name, fid))
                param_list[-1].append((wh, bh))
            wout = randmat((self.dim_h, dimin), name='w{}_{}_{}'.format(self.n_hidden, self.name, fid))
            bout = tf.Variable(tf.zeros((dimin,)), name='b{}_{}_{}'.format(self.n_hidden, self.name, fid))
            wout2 = randmat((self.dim_h, dimin), name='w{}_{}_{}_sigma'.format(self.n_hidden, self.name, fid))
            bout2 = tf.Variable(tf.ones((dimin,)) * 2, name='b{}_{}_{}_sigma'.format(self.n_hidden, self.name, fid))
            param_list[-1].append((wout, bout, wout2, bout2))

    def build(self):
        for flow in xrange(self.n_flows):
            self.build_mnn('muf_{}'.format(flow), self.params)

    def ff(self, x, weights):
        inputs = [x]
        for j in xrange(len(weights[:-1])):
            h = tf.matmul(inputs[-1], weights[j][0]) + weights[j][1]
            inputs.append(self.nonlin(h))
        wmu, bmu, wsigma, bsigma = weights[-1]
        mean = tf.matmul(inputs[-1], wmu) + bmu
        sigma = tf.matmul(inputs[-1], wsigma) + bsigma
        return mean, sigma

    def get_output_for(self, z, sample=True):
        logdets = zeros_d((tf.shape(z)[0],))
        for flow in xrange(self.n_flows):
            mask = random_bernoulli(tf.shape(z), p=0.5) if sample else 0.5
            ggmu, ggsigma = self.ff(mask * z, self.params[flow])
            gate = tf.nn.sigmoid(ggsigma)
            logdets += tf.reduce_sum((1 - mask) * tf.log(gate), axis=1)
            z = (1 - mask) * (z * gate + (1 - gate) * ggmu) + mask * z

        return z, logdets


class PlanarFlow(object):
    """
    """
    def __init__(self, incoming, n_flows=2, name=None, scope=None, **kwargs):
        self.incoming = incoming
        self.n_flows = n_flows
        self.sigma = 0.01
        self.params = []
        self.name = name
        self.scope = scope
        self.build()
        print 'Planar flow layer with nf: {}, name: {}, scope: {}'.format(n_flows, name, scope)

    def build(self):
        with tf.variable_scope(self.scope):
            for flow in xrange(self.n_flows):
                w = randmat((self.incoming, 1), name='w_{}_{}'.format(flow, self.name))
                u = randmat((self.incoming, 1), name='u_{}_{}'.format(flow, self.name))
                b = tf.Variable(tf.zeros((1,)), name='b_{}_{}'.format(flow, self.name))
                self.params.append([w, u, b])

    def get_output_for(self, z, **kwargs):
        logdets = zeros_d((tf.shape(z)[0],))
        for flow in xrange(self.n_flows):
            w, u, b = self.params[flow]
            uw = tf.reduce_sum(u * w)
            muw = -1 + tf.nn.softplus(uw)  # = -1 + T.log(1 + T.exp(uw))
            u_hat = u + (muw - uw) * w / tf.reduce_sum(w ** 2)
            if len(z.get_shape()) == 1:
                zwb = z * w + b
            else:
                zwb = tf.matmul(z, w) + b
            psi = tf.matmul(1 - tf.nn.tanh(zwb) ** 2, tf.transpose(w))  # tanh(x)dx = 1 - tanh(x)**2
            psi_u = tf.matmul(psi, u_hat)
            logdets += tf.squeeze(tf.log(tf.abs(1 + psi_u)))
            zadd = tf.matmul(tf.nn.tanh(zwb), tf.transpose(u_hat))
            z += zadd
        return z, logdets