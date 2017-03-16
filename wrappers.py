from layers import *
from tensorflow.contrib import slim


class MNFLeNet(object):
    def __init__(self, N, input_shape, flows_q=2, flows_r=2, use_z=True,  activation=tf.nn.relu, logging=False,
                 nb_classes=10, learn_p=False, layer_dims=(20, 50, 500), flow_dim_h=50, thres_var=1, prior_var_w=1.,
                 prior_var_b=1.):
        self.layer_dims = layer_dims
        self.activation = activation
        self.N = N
        self.input_shape = input_shape
        self.flows_q = flows_q
        self.flows_r = flows_r
        self.use_z = use_z

        self.logging = logging
        self.nb_classes = nb_classes
        self.flow_dim_h = flow_dim_h
        self.thres_var = thres_var
        self.learn_p = learn_p
        self.prior_var_w = prior_var_w
        self.prior_var_b = prior_var_b

        self.opts = 'fq{}_fr{}_usez{}'.format(self.flows_q, self.flows_r, self.use_z)
        self.built = False

    def build_mnf_lenet(self, x, sample=True):
        if not self.built:
            self.layers = []
        with tf.variable_scope(self.opts):
            if not self.built:
                layer1 = Conv2DMNF(self.layer_dims[0], 5, 5, N=self.N, input_shape=self.input_shape, border_mode='VALID',
                                   flows_q=self.flows_q, flows_r=self.flows_r, logging=self.logging, use_z=self.use_z,
                                   learn_p=self.learn_p, prior_var=self.prior_var_w, prior_var_b=self.prior_var_b,
                                   thres_var=self.thres_var, flow_dim_h=self.flow_dim_h)
                self.layers.append(layer1)
            else:
                layer1 = self.layers[0]
            h1 = self.activation(tf.nn.max_pool(layer1(x, sample=sample), [1, 2, 2, 1], [1, 2, 2, 1], 'SAME'))

            if not self.built:
                shape = [None] + [s.value for s in h1.get_shape()[1:]]
                layer2 = Conv2DMNF(self.layer_dims[1], 5, 5, N=self.N, input_shape=shape, border_mode='VALID',
                                   flows_q=self.flows_q, flows_r=self.flows_r, use_z=self.use_z, logging=self.logging,
                                   learn_p=self.learn_p, flow_dim_h=self.flow_dim_h, thres_var=self.thres_var,
                                   prior_var=self.prior_var_w, prior_var_b=self.prior_var_b)
                self.layers.append(layer2)
            else:
                layer2 = self.layers[1]
            h2 = slim.flatten(self.activation(tf.nn.max_pool(layer2(h1, sample=sample), [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')))

            if not self.built:
                fcinp_dim = h2.get_shape()[1].value
                layer3 = DenseMNF(self.layer_dims[2], N=self.N, input_dim=fcinp_dim, flows_q=self.flows_q,
                                  flows_r=self.flows_r, use_z=self.use_z, logging=self.logging, learn_p=self.learn_p,
                                  prior_var=self.prior_var_w, prior_var_b=self.prior_var_b, flow_dim_h=self.flow_dim_h,
                                  thres_var=self.thres_var)
                self.layers.append(layer3)
            else:
                layer3 = self.layers[2]
            h3 = self.activation(layer3(h2, sample=sample))

            if not self.built:
                fcinp_dim = h3.get_shape()[1].value
                layerout = DenseMNF(self.nb_classes, N=self.N, input_dim=fcinp_dim, flows_q=self.flows_q,
                                    flows_r=self.flows_r, use_z=self.use_z, logging=self.logging, learn_p=self.learn_p,
                                    prior_var=self.prior_var_w, prior_var_b=self.prior_var_b, flow_dim_h=self.flow_dim_h,
                                    thres_var=self.thres_var)
                self.layers.append(layerout)
            else:
                layerout = self.layers[3]

        if not self.built:
            self.built = True
        return layerout(h3, sample=sample)

    def predict(self, x, sample=True):
        return self.build_mnf_lenet(x, sample=sample)

    def get_reg(self):
        reg = 0.
        for j, layer in enumerate(self.layers):
            with tf.name_scope('kl_layer{}'.format(j + 1)):
                regi = layer.get_reg()
                tf.summary.scalar('kl_layer{}'.format(j + 1), regi)
            reg += regi

        return reg


