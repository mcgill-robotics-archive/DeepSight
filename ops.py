
import tensorflow as tf

BN_EPSILON = 0.001

def kernel_variable(name, shape, dtype=tf.float32, trainable=True, collection=tf.GraphKeys.GLOBAL_VARIABLES):
    return tf.get_variable(name,
                           initializer=tf.truncated_normal(shape=shape, mean=0.0, stddev=1e-1, dtype=dtype),
                           dtype=dtype,
                           trainable=trainable,
                           collections=[collection])


def bias_variable(name, shape, dtype=tf.float32, trainable=True, collection=tf.GraphKeys.GLOBAL_VARIABLES):
    return tf.get_variable(name,
                           initializer=tf.zeros(shape=shape, dtype=dtype),
                           trainable=trainable,
                           collections=[collection])


def instance_norm(x, gamma=None, beta=None, trainable=True, collection=tf.GraphKeys.GLOBAL_VARIABLES):

    params_shape = x.get_shape()[-1:]
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

    if gamma is None:
        gamma = tf.get_variable('gamma',
                                initializer=tf.ones(shape=params_shape, dtype=tf.float32),
                                trainable=trainable,
                                dtype=tf.float32,
                                collections=[collection])

    if beta is None:
        beta = tf.get_variable('beta',
                               initializer=tf.zeros(shape=params_shape, dtype=tf.float32),
                               trainable=trainable,
                               dtype=tf.float32,
                               collections=[collection])

    return tf.nn.batch_normalization(x, mean, var, beta, gamma, BN_EPSILON)


def max_pool(x, pool_size, strides):
    return tf.nn.max_pool(x,
                          ksize=[1] + list(pool_size) + [1],
                          strides=[1] + list(strides) + [1],
                          padding='SAME',
                          name='max_pool')


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

    scope_name = var.name.split(':')[0]
    with tf.name_scope(scope_name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def conv_layer(x, filter_size, num_features, strides, trainable=True,
               relu=True, deconv=False, upscale=2,
               mirror_pad=True, collection=tf.GraphKeys.GLOBAL_VARIABLES):
    x_shape = x.get_shape().as_list()

    filters_in = x_shape[-1]

    if deconv:
        kernel_shape = list(filter_size) + [num_features, filters_in]
    else:
        kernel_shape = list(filter_size) + [filters_in, num_features]

    weights = kernel_variable(name='weights',
                              shape=kernel_shape,
                              trainable=trainable,
                              collection=collection)
    biases = bias_variable(name='biases',
                           shape=[num_features],
                           trainable=trainable,
                           collection=collection)

    if deconv:

        shape = tf.shape(x)
        num_features_tensor = tf.constant(num_features, dtype=tf.int32)
        try:
            output_size = tf.pack([shape[0], shape[1] * upscale, shape[2] * upscale, num_features_tensor], axis=0)
        except AttributeError:
            output_size = tf.stack([shape[0], shape[1] * upscale, shape[2] * upscale, num_features_tensor], axis=0)

        conv = tf.nn.conv2d_transpose(x, weights,
                                      strides=[1] + list(strides) + [1],
                                      output_shape=output_size,
                                      padding='SAME',
                                      name='deconv')
    else:
        # Implement mirror padding (removes border around images)
        if mirror_pad:
            padding = 'VALID'

            # Mirror padding
            pad_amount = filter_size[0] // 2
            x = tf.pad(
                x, [[0, 0], [pad_amount, pad_amount], [pad_amount, pad_amount], [0, 0]],
                mode='REFLECT')
        else:
            padding = 'SAME'

        conv = tf.nn.conv2d(x, weights,
                            strides=[1] + list(strides) + [1],
                            padding=padding,
                            name='conv')

    bias = tf.nn.bias_add(conv, biases, name='bias_add')

    if relu:
        return tf.nn.relu(bias, name='activations')
    else:
        return bias
