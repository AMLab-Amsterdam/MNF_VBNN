import tensorflow as tf
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
from keras.utils.np_utils import to_categorical
from mnist import MNIST
import time, os
from wrappers import MNFLeNet


def train():
    mnist = MNIST()
    (xtrain, ytrain), (xvalid, yvalid), (xtest, ytest) = mnist.images()
    xtrain, xvalid, xtest = np.transpose(xtrain, [0, 2, 3, 1]), np.transpose(xvalid, [0, 2, 3, 1]), np.transpose(xtest, [0, 2, 3, 1])
    ytrain, yvalid, ytest = to_categorical(ytrain, 10), to_categorical(yvalid, 10), to_categorical(ytest, 10)

    N, height, width, n_channels = xtrain.shape
    iter_per_epoch = N / 100

    sess = tf.InteractiveSession()

    input_shape = [None, height, width, n_channels]
    x = tf.placeholder(tf.float32, input_shape, name='x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

    model = MNFLeNet(N, input_shape=input_shape, flows_q=FLAGS.fq, flows_r=FLAGS.fr, use_z=not FLAGS.no_z,
                     learn_p=FLAGS.learn_p, thres_var=FLAGS.thres_var, flow_dim_h=FLAGS.flow_h)

    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    y = model.predict(x)
    yd = model.predict(x, sample=False)
    pyx = tf.nn.softmax(y)

    with tf.name_scope('KL_prior'):
        regs = model.get_reg()
        tf.summary.scalar('KL prior', regs)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        tf.summary.scalar('Loglike', cross_entropy)

    global_step = tf.Variable(0, trainable=False)
    if FLAGS.anneal:
        number_zero, original_zero = FLAGS.epzero, FLAGS.epochs / 2
        with tf.name_scope('annealing_beta'):
            max_zero_step = number_zero * iter_per_epoch
            original_anneal = original_zero * iter_per_epoch
            beta_t_val = tf.cast((tf.cast(global_step, tf.float32) - max_zero_step) / original_anneal, tf.float32)
            beta_t = tf.maximum(beta_t_val, 0.)
            annealing = tf.minimum(1., tf.cond(global_step < max_zero_step, lambda: tf.zeros((1,))[0], lambda: beta_t))
            tf.summary.scalar('annealing beta', annealing)
    else:
        annealing = 1.

    with tf.name_scope('lower_bound'):
        lowerbound = cross_entropy + annealing * regs
        tf.summary.scalar('Lower bound', lowerbound)

    train_step = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(lowerbound, global_step=global_step)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(yd, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)

    tf.add_to_collection('logits', y)
    tf.add_to_collection('logits_map', yd)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y_)
    saver = tf.train.Saver(tf.global_variables())

    tf.global_variables_initializer().run()

    idx = np.arange(N)
    steps = 0
    model_dir = './models/mnf_lenet_mnist_fq{}_fr{}_usez{}_thres{}/model/'.format(FLAGS.fq, FLAGS.fr, not FLAGS.no_z,
                                                                                  FLAGS.thres_var)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print 'Will save model as: {}'.format(model_dir + 'model')
    # Train
    for epoch in xrange(FLAGS.epochs):
        widgets = ["epoch {}/{}|".format(epoch + 1, FLAGS.epochs), Percentage(), Bar(), ETA()]
        pbar = ProgressBar(iter_per_epoch, widgets=widgets)
        pbar.start()
        np.random.shuffle(idx)
        t0 = time.time()
        for j in xrange(iter_per_epoch):
            steps += 1
            pbar.update(j)
            batch = np.random.choice(idx, 100)
            if j == (iter_per_epoch - 1):
                summary, _ = sess.run([merged, train_step], feed_dict={x: xtrain[batch], y_: ytrain[batch]})
                train_writer.add_summary(summary,  steps)
                train_writer.flush()
            else:
                sess.run(train_step, feed_dict={x: xtrain[batch], y_: ytrain[batch]})

        # the accuracy here is calculated by a crude MAP so as to have fast evaluation
        # it is much better if we properly integrate over the parameters by averaging across multiple samples
        tacc = sess.run(accuracy, feed_dict={x: xvalid, y_: yvalid})
        string = 'Epoch {}/{}, valid_acc: {:0.3f}'.format(epoch + 1, FLAGS.epochs, tacc)

        if (epoch + 1) % 10 == 0:
            string += ', model_save: True'
            saver.save(sess, model_dir + 'model')

        string += ', dt: {:0.3f}'.format(time.time() - t0)
        print string

    saver.save(sess, model_dir + 'model')
    train_writer.close()

    preds = np.zeros_like(ytest)
    widgets = ["Sampling |", Percentage(), Bar(), ETA()]
    pbar = ProgressBar(FLAGS.L, widgets=widgets)
    pbar.start()
    for i in xrange(FLAGS.L):
        pbar.update(i)
        for j in xrange(xtest.shape[0] / 100):
            pyxi = sess.run(pyx, feed_dict={x: xtest[j * 100:(j + 1) * 100]})
            preds[j * 100:(j + 1) * 100] += pyxi / FLAGS.L
    print
    sample_accuracy = np.mean(np.equal(np.argmax(preds, 1), np.argmax(ytest, 1)))
    print 'Sample test accuracy: {}'.format(sample_accuracy)


def main():
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries_dir', type=str, default='logs/mnf_lenet',
                        help='Summaries directory')
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-epzero', type=int, default=1)
    parser.add_argument('-fq', default=2, type=int)
    parser.add_argument('-fr', default=2, type=int)
    parser.add_argument('-no_z', action='store_true')
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-thres_var', type=float, default=0.5)
    parser.add_argument('-flow_h', type=int, default=50)
    parser.add_argument('-L', type=int, default=100)
    parser.add_argument('-anneal', action='store_true')
    parser.add_argument('-learn_p', action='store_true')
    FLAGS = parser.parse_args()
    main()
