import tensorflow as tf


def weight_variable(shape, name=None):
    with tf.name_scope('weight'):
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.0001))
        tf.histogram_summary('%s/weight' % (name if name is not None else ''), weight)
        return weight


def bias_variable(shape, name=None):
    with tf.name_scope('bias'):
        bias = tf.Variable(tf.zeros(shape=shape))
        tf.histogram_summary('%s/bias' % (name if name is not None else ''), bias)
        return bias


def ff_layer(input_layer, depth, name = None):
    """ Constructs a fully connected layer which takes input_layer as input.


    :param input_layer:
    :param depth: Number of output nodes.
    :param name: Name of layer.
    """
    with tf.name_scope(name):
        assert(input_layer.get_shape().ndims == 2)
        w = weight_variable([input_layer.get_shape().as_list()[-1], depth], name)
        b = bias_variable([depth], name)
        output = tf.matmul(input_layer, w) + b
        return output