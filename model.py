from utils import (
    PS1
)
import tensorflow as tf


def define_model(x_in, x_bic):

    def create_variable(shape):
        """
        Create a convolution filter variable with the specified name and shape,
        and initialize it using Xavier initialition.
        """
        # initializer = tf.contrib.layers.variance_scaling_initializer()
        # variable = tf.Variable(initializer(shape=shape), dtype=tf.float32)
        # tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.001)(initializer))
        variable = tf.Variable(tf.truncated_normal(shape, stddev=0.001, dtype=tf.float32))
        # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.001)(var))
        return variable

    def create_bias_variable(shape):
        """Create a bias variable with the specified name and shape and initialize
        it to zero."""
        initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)
        return tf.Variable(initializer(shape=shape))

    def conv2d_same(x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    with tf.variable_scope("conv1"):
        w1 = create_bias_variable([3, 3, 1, 64])
        b1 = create_variable([64])
        conv1 = tf.nn.relu(conv2d_same(x_in, w1) + b1)

    with tf.variable_scope("RBF1"):
        w2 = create_variable([3, 3, 64, 64])
        w3 = create_variable([3, 3, 64, 64])
        w4 = create_variable([3, 3, 64, 64])
        w5 = create_variable([3, 3, 64, 64])
        w6 = create_variable([1, 1, 256, 64])

        b2 = create_bias_variable([64])
        b3 = create_bias_variable([64])
        b4 = create_bias_variable([64])
        b5 = create_bias_variable([64])
        b6 = create_bias_variable([64])

        # first RFB model
        input_1 = conv1
        conv2 = tf.nn.relu(conv2d_same(input_1, w2) + b2)

        conv3 = tf.nn.relu(conv2d_same(conv2 + conv1, w3) + b3)

        conv4 = tf.nn.relu(conv2d_same(conv3 + conv2, w4) + b4)

        conv5 = tf.nn.relu(conv2d_same(conv4 + conv3, w5) + b5)

        concat_1 = tf.concat([conv2, conv3, conv4, conv5], 3)

        conv6 = tf.nn.relu(conv2d_same(concat_1, w6) + b6)

        out_1 = conv6

    with tf.variable_scope("RBF2"):
        w2 = create_variable([3, 3, 64, 64])
        w3 = create_variable([3, 3, 64, 64])
        w4 = create_variable([3, 3, 64, 64])
        w5 = create_variable([3, 3, 64, 64])
        w6 = create_variable([1, 1, 256, 64])

        b2 = create_bias_variable([64])
        b3 = create_bias_variable([64])
        b4 = create_bias_variable([64])
        b5 = create_bias_variable([64])
        b6 = create_bias_variable([64])

        # second RFB model
        input_1 = conv1 + out_1
        conv2 = tf.nn.relu(conv2d_same(input_1, w2) + b2)

        conv3 = tf.nn.relu(conv2d_same(conv2 + conv1, w3) + b3)

        conv4 = tf.nn.relu(conv2d_same(conv3 + conv2, w4) + b4)

        conv5 = tf.nn.relu(conv2d_same(conv4 + conv3, w5) + b5)

        concat_1 = tf.concat([conv2, conv3, conv4, conv5], 3)

        conv6 = tf.nn.relu(conv2d_same(concat_1, w6) + b6)

        out_2 = conv6

    with tf.variable_scope("RBF3"):
        w2 = create_variable([3, 3, 64, 64])
        w3 = create_variable([3, 3, 64, 64])
        w4 = create_variable([3, 3, 64, 64])
        w5 = create_variable([3, 3, 64, 64])
        w6 = create_variable([1, 1, 256, 64])

        b2 = create_bias_variable([64])
        b3 = create_bias_variable([64])
        b4 = create_bias_variable([64])
        b5 = create_bias_variable([64])
        b6 = create_bias_variable([64])

        # third RFB model
        input_1 = conv1 + out_1 + out_2
        conv2 = tf.nn.relu(conv2d_same(input_1, w2) + b2)

        conv3 = tf.nn.relu(conv2d_same(conv2 + conv1, w3) + b3)

        conv4 = tf.nn.relu(conv2d_same(conv3 + conv2, w4) + b4)

        conv5 = tf.nn.relu(conv2d_same(conv4 + conv3, w5) + b5)

        concat_1 = tf.concat([conv2, conv3, conv4, conv5], 3)

        conv6 = tf.nn.relu(conv2d_same(concat_1, w6) + b6)

        out_3 = conv6

    with tf.variable_scope("RBF4"):
        w2 = create_variable([3, 3, 64, 64])
        w3 = create_variable([3, 3, 64, 64])
        w4 = create_variable([3, 3, 64, 64])
        w5 = create_variable([3, 3, 64, 64])
        w6 = create_variable([1, 1, 256, 64])

        b2 = create_bias_variable([64])
        b3 = create_bias_variable([64])
        b4 = create_bias_variable([64])
        b5 = create_bias_variable([64])
        b6 = create_bias_variable([64])

        # forth RFB model
        input_1 = conv1 + out_1 + out_2 + out_3
        conv2 = tf.nn.relu(conv2d_same(input_1, w2) + b2)

        conv3 = tf.nn.relu(conv2d_same(conv2 + conv1, w3) + b3)

        conv4 = tf.nn.relu(conv2d_same(conv3 + conv2, w4) + b4)

        conv5 = tf.nn.relu(conv2d_same(conv4 + conv3, w5) + b5)

        concat_1 = tf.concat([conv2, conv3, conv4, conv5], 3)

        conv6 = tf.nn.relu(conv2d_same(concat_1, w6) + b6)

        out_4 = conv6

    with tf.variable_scope("RBF4"):
        w_up = create_variable([3, 3, 64, 4])
        b_up = create_bias_variable([4])
        conv_up = tf.nn.relu(conv2d_same(out_4, w_up) + b_up)

        out_nn = PS1(conv_up, 2, 1)

        return out_nn + x_bic
