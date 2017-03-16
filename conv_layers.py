from layers import Layer
import numpy as np
import tensorflow as tf
from norm_flows import MaskedNVPFlow
from utils import randmat, zeros_d, ones_d, outer


class Conv2DMNF(Layer):
    '''2D convolutional layer with a multiplicative normalizing flow (MNF) aproximate posterior over the weights.
    Prior is a standard normal.
    '''

    def __init__(self, nb_filter, nb_row, nb_col, input_shape=(), activation=tf.identity, N=1, name=None,
                 border_mode='SAME', subsample=(1, 1, 1, 1), flows_q=2, flows_r=2, learn_p=False, use_z=True,
                 prior_var=1., prior_var_b=1., flow_dim_h=50, logging=False,  thres_var=1., **kwargs):

        if border_mode not in {'VALID', 'SAME'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)

        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.border_mode = border_mode
        self.subsample = subsample
        self.thres_var = thres_var

        self.N = N
        self.flow_dim_h = flow_dim_h
        self.learn_p = learn_p
        self.input_shape = input_shape

        self.prior_var = prior_var
        self.prior_var_b = prior_var_b
        self.n_flows_q = flows_q
        self.n_flows_r = flows_r
        self.use_z = use_z
        super(Conv2DMNF, self).__init__(N=N, nonlin=activation, name=name, logging=logging)

    def build(self):
        stack_size = self.input_shape[-1]
        self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
        self.input_dim = self.nb_col * stack_size * self.nb_row
        self.stack_size = stack_size

        with tf.variable_scope(self.name):
            self.mu_W = randmat(self.W_shape, name='mean_W')
            self.logvar_W = randmat(self.W_shape, mu=-9., name='logvar_W', extra_scale=1e-6)
            self.mu_bias = tf.Variable(tf.zeros((self.nb_filter,)), name='mean_bias')
            self.logvar_bias = randmat((self.nb_filter,), mu=-9., name='logvar_bias', extra_scale=1e-6)

            if self.use_z:
                self.qzero_mean = randmat((self.nb_filter,), name='dropout_rates_mean', mu=1. if self.n_flows_q == 0 else 0.)
                self.qzero = randmat((self.nb_filter,), name='dropout_rates', mu=np.log(0.1), extra_scale=1e-6)
                self.rsr_M = randmat((self.nb_filter,), name='var_r_aux')
                self.apvar_M = randmat((self.nb_filter,), name='apvar_r_aux')
                self.rsri_M = randmat((self.nb_filter,), name='var_r_auxi')

            self.pvar = randmat((self.input_dim,), mu=np.log(self.prior_var), name='prior_var_r_p', extra_scale=1e-6, trainable=self.learn_p)
            self.pvar_bias = randmat((1,), mu=np.log(self.prior_var_b), name='prior_var_r_p_bias', extra_scale=1e-6, trainable=self.learn_p)

        if self.n_flows_r > 0:
            self.flow_r = MaskedNVPFlow(self.nb_filter, n_flows=self.n_flows_r, name=self.name + '_fr', n_hidden=0,
                                        dim_h=2 * self.flow_dim_h, scope=self.name)

        if self.n_flows_q > 0:
            self.flow_q = MaskedNVPFlow(self.nb_filter, n_flows=self.n_flows_q, name=self.name + '_fq', n_hidden=0,
                                        dim_h=self.flow_dim_h, scope=self.name)

        print 'Built layer {}, output_dim: {}, input_shape: {}, flows_r: {}, flows_q: {}, use_z: {}, learn_p: {}, ' \
              'pvar: {}, thres_var: {}'.format(self.name, self.nb_filter, self.input_shape, self.n_flows_r,
                                               self.n_flows_q, self.use_z, self.learn_p, self.prior_var, self.thres_var)

    def sample_z(self, size_M=1, sample=True):
        if not self.use_z:
            return ones_d((size_M, self.nb_filter)), zeros_d((size_M,))
        qm0 = self.get_params_m()
        isample_M = tf.tile(tf.expand_dims(self.qzero_mean, 0), [size_M, 1])
        eps = tf.random_normal(tf.stack((size_M, self.nb_filter)))
        sample_M = isample_M + tf.sqrt(qm0) * eps if sample else isample_M

        logdets = zeros_d((size_M,))
        if self.n_flows_q > 0:
            sample_M, logdets = self.flow_q.get_output_for(sample_M, sample=sample)

        return sample_M, logdets

    def get_params_m(self):
        if not self.use_z:
            return None

        return tf.exp(self.qzero)

    def get_params_W(self):
        return tf.exp(self.logvar_W)

    def get_mean_var(self, x):
        var_w = tf.clip_by_value(self.get_params_W(), 0., self.thres_var)
        var_w = tf.square(var_w)
        var_b = tf.clip_by_value(tf.exp(self.logvar_bias), 0., self.thres_var**2)

        # formally we do cross-correlation here
        muout = tf.nn.conv2d(x, self.mu_W, self.subsample, self.border_mode, use_cudnn_on_gpu=True) + self.mu_bias
        varout = tf.nn.conv2d(tf.square(x), var_w, self.subsample, self.border_mode, use_cudnn_on_gpu=True) + var_b

        return muout, varout

    def kldiv(self):
        M, logdets = self.sample_z()
        logdets = logdets[0]
        M = tf.squeeze(M)

        std_w = self.get_params_W()
        mu = tf.reshape(self.mu_W, [-1, self.nb_filter])
        std_w = tf.reshape(std_w, [-1, self.nb_filter])
        Mtilde = mu * tf.expand_dims(M, 0)
        mbias = self.mu_bias * M
        Vtilde = tf.square(std_w)

        iUp = outer(tf.exp(self.pvar), ones_d((self.nb_filter,)))

        qm0 = self.get_params_m()
        logqm = 0.
        if self.use_z > 0.:
            logqm = - tf.reduce_sum(.5 * (tf.log(2 * np.pi) + tf.log(qm0) + 1))
            logqm -= logdets

        kldiv_w = tf.reduce_sum(.5 * tf.log(iUp) - .5 * tf.log(Vtilde) + ((Vtilde + tf.square(Mtilde)) / (2 * iUp)) - .5)
        kldiv_bias = tf.reduce_sum(.5 * self.pvar_bias - .5 * self.logvar_bias + ((tf.exp(self.logvar_bias) +
                                                                                   tf.square(mbias)) / (2 * tf.exp(self.pvar_bias))) - .5)

        logrm = 0.
        if self.use_z:
            apvar_M = self.apvar_M
            mw = tf.matmul(Mtilde, tf.expand_dims(apvar_M, 1))
            vw = tf.matmul(Vtilde, tf.expand_dims(tf.square(apvar_M), 1))
            eps = tf.expand_dims(tf.random_normal((self.input_dim,)), 1)
            a = mw + tf.sqrt(vw) * eps
            mb = tf.reduce_sum(mbias * apvar_M)
            vb = tf.reduce_sum(tf.exp(self.logvar_bias) * tf.square(apvar_M))
            a += mb + tf.sqrt(vb) * tf.random_normal(())

            w__ = tf.reduce_mean(outer(tf.squeeze(a), self.rsr_M), axis=0)
            wv__ = tf.reduce_mean(outer(tf.squeeze(a), self.rsri_M), axis=0)

            if self.flow_r is not None:
                M, logrm = self.flow_r.get_output_for(tf.expand_dims(M, 0))
                M = tf.squeeze(M)
                logrm = logrm[0]

            logrm += tf.reduce_sum(-.5 * tf.exp(wv__) * tf.square(M - w__) - .5 * tf.log(2 * np.pi) + .5 * wv__)

        return - kldiv_w + logrm - logqm - kldiv_bias

    def call(self, x, sample=True, **kwargs):
        sample_M, _ = self.sample_z(size_M=tf.shape(x)[0], sample=sample)
        sample_M = tf.expand_dims(tf.expand_dims(sample_M, 1), 2)
        mean_out, var_out = self.get_mean_var(x)
        mean_gout = mean_out * sample_M
        var_gout = tf.sqrt(var_out) * tf.random_normal(tf.shape(mean_gout))
        out = mean_gout + var_gout

        output = out if sample else mean_gout
        return output
