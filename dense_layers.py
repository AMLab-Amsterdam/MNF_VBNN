from layers import Layer
import numpy as np
import tensorflow as tf
from norm_flows import MaskedNVPFlow, PlanarFlow
from utils import randmat, zeros_d, ones_d, outer


class DenseMNF(Layer):
    '''Fully connected layer with a multiplicative normalizing flow (MNF) aproximate posterior over the weights.
    Prior is a standard normal.
    '''
    def __init__(self, output_dim, activation=tf.identity, N=1, input_dim=None, flows_q=2, flows_r=2, learn_p=False,
                 use_z=True, prior_var=1., name=None, logging=False, flow_dim_h=50, prior_var_b=1., thres_var=1.,
                 **kwargs):

        self.output_dim = output_dim
        self.learn_p = learn_p
        self.prior_var = prior_var
        self.prior_var_b = prior_var_b
        self.thres_var = thres_var

        self.n_flows_q = flows_q
        self.n_flows_r = flows_r
        self.use_z = use_z
        self.flow_dim_h = flow_dim_h

        self.input_dim = input_dim
        super(DenseMNF, self).__init__(N=N, nonlin=activation, name=name, logging=logging)

    def build(self):
        dim_in, dim_out = self.input_dim, self.output_dim

        with tf.variable_scope(self.name):
            self.mu_W = randmat((dim_in, dim_out), name='mean_W', extra_scale=1.)
            self.logvar_W = randmat((dim_in, dim_out), mu=-9., name='var_W', extra_scale=1e-6)
            self.mu_bias = tf.Variable(tf.zeros((dim_out,)), name='mean_bias')
            self.logvar_bias = randmat((dim_out,), mu=-9., name='var_bias', extra_scale=1e-6)

            if self.use_z:
                self.qzero_mean = randmat((dim_in,), name='dropout_rates_mean', mu=1. if self.n_flows_q == 0 else 0.)
                self.qzero = randmat((dim_in,), mu=np.log(0.1), name='dropout_rates', extra_scale=1e-6)
                self.rsr_M = randmat((dim_in,), name='var_r_aux')
                self.apvar_M = randmat((dim_in,), name='apvar_r_aux')
                self.rsri_M = randmat((dim_in,), name='var_r_auxi')

            self.pvar = randmat((dim_in,), mu=np.log(self.prior_var), name='prior_var_r_p', trainable=self.learn_p, extra_scale=1e-6)
            self.pvar_bias = randmat((1,), mu=np.log(self.prior_var_b), name='prior_var_r_p_bias', trainable=self.learn_p, extra_scale=1e-6)

        if self.n_flows_r > 0:
            if dim_in == 1:
                self.flow_r = PlanarFlow(dim_in, n_flows=self.n_flows_r, name=self.name + '_fr', scope=self.name)
            else:
                self.flow_r = MaskedNVPFlow(dim_in, n_flows=self.n_flows_r, name=self.name + '_fr', n_hidden=0,
                                            dim_h=2 * self.flow_dim_h, scope=self.name)

        if self.n_flows_q > 0:
            if dim_in == 1:
                self.flow_q = PlanarFlow(dim_in, n_flows=self.n_flows_q, name=self.name + '_fq', scope=self.name)
            else:
                self.flow_q = MaskedNVPFlow(dim_in, n_flows=self.n_flows_q, name=self.name + '_fq', n_hidden=0,
                                            dim_h=self.flow_dim_h,  scope=self.name)

        print 'Built layer', self.name, 'prior_var: {}'.format(self.prior_var), \
            'flows_q: {}, flows_r: {}, use_z: {}'.format(self.n_flows_q, self.n_flows_r, self.use_z), \
            'learn_p: {}, thres_var: {}'.format(self.learn_p, self.thres_var)

    def sample_z(self, size_M=1, sample=True):
        if not self.use_z:
            return ones_d((size_M, self.input_dim)), zeros_d((size_M,))

        qm0 = self.get_params_m()
        isample_M = tf.tile(tf.expand_dims(self.qzero_mean, 0), [size_M, 1])
        eps = tf.random_normal(tf.stack((size_M, self.input_dim)))
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

    def kldiv(self):
        M, logdets = self.sample_z()
        logdets = logdets[0]
        M = tf.squeeze(M)

        std_mg = self.get_params_W()
        qm0 = self.get_params_m()
        if len(M.get_shape()) == 0:
            Mexp = M
        else:
            Mexp = tf.expand_dims(M, 1)

        Mtilde = Mexp * self.mu_W
        Vtilde = tf.square(std_mg)

        iUp = outer(tf.exp(self.pvar), ones_d((self.output_dim,)))

        logqm = 0.
        if self.use_z:
            logqm = - tf.reduce_sum(.5 * (tf.log(2 * np.pi) + tf.log(qm0) + 1))
            logqm -= logdets

        kldiv_w = tf.reduce_sum(.5 * tf.log(iUp) - tf.log(std_mg) + ((Vtilde + tf.square(Mtilde)) / (2 * iUp)) - .5)
        kldiv_bias = tf.reduce_sum(.5 * self.pvar_bias - .5 * self.logvar_bias + ((tf.exp(self.logvar_bias) +
                                                                                   tf.square(self.mu_bias)) / (2 * tf.exp(self.pvar_bias))) - .5)

        if self.use_z:
            apvar_M = self.apvar_M
            # shared network for hidden layer
            mw = tf.matmul(tf.expand_dims(apvar_M, 0), Mtilde)
            eps = tf.expand_dims(tf.random_normal((self.output_dim,)), 0)
            varw = tf.matmul(tf.square(tf.expand_dims(apvar_M, 0)), Vtilde)
            a = tf.nn.tanh(mw + tf.sqrt(varw) * eps)
            # split at output layer
            if len(tf.squeeze(a).get_shape()) != 0:
                w__ = tf.reduce_mean(outer(self.rsr_M, tf.squeeze(a)), axis=1)
                wv__ = tf.reduce_mean(outer(self.rsri_M, tf.squeeze(a)), axis=1)
            else:
                w__ = self.rsr_M * tf.squeeze(a)
                wv__ = self.rsri_M * tf.squeeze(a)

            logrm = 0.
            if self.flow_r is not None:
                M, logrm = self.flow_r.get_output_for(tf.expand_dims(M, 0))
                M = tf.squeeze(M)
                logrm = logrm[0]

            logrm += tf.reduce_sum(-.5 * tf.exp(wv__) * tf.square(M - w__) - .5 * tf.log(2 * np.pi) + .5 * wv__)
        else:
            logrm = 0.

        return - kldiv_w + logrm - logqm - kldiv_bias

    def call(self, x, sample=True, **kwargs):
        std_mg = tf.clip_by_value(self.get_params_W(), 0., self.thres_var)
        var_mg = tf.square(std_mg)
        sample_M, _ = self.sample_z(size_M=tf.shape(x)[0], sample=sample)
        xt = x * sample_M

        mu_out = tf.matmul(xt, self.mu_W) + self.mu_bias
        varin = tf.matmul(tf.square(x), var_mg) + tf.clip_by_value(tf.exp(self.logvar_bias), 0., self.thres_var**2)
        xin = tf.sqrt(varin)
        sigma_out = xin * tf.random_normal(tf.shape(mu_out))

        output = mu_out + sigma_out if sample else mu_out
        return output

