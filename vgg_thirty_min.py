
import tensorflow as tf
import numpy as np
import sys

from DataSet import create_data_sets

NUM_EPOCHS = 1
LEARNING_RATE = 0.001

# Creates a variable initialized by a normal distribution
def _kernel_variable(shape, name, trainable):
    return tf.Variable(tf.truncated_normal(shape, dtype=tf.float32, stddev=1e-1), name=name, trainable=trainable)

# Creates a bias variable initialized by 0s
def _bias_variable(shape, name, trainable):
    return tf.Variable(tf.constant(0.0, shape=shape, dtype=tf.float32), trainable=trainable, name=name)

# Loads weights from an NPZ into a dictionary of tf variables
def _load_weights(variable_dict, weight_file, sess):
    weights = np.load(weight_file)
    keys = sorted(weights.keys())

    for k in keys:
        if k in variable_dict:
            print "Layer %s, shape:" % k, np.shape(weights[k])
            sess.run(variable_dict[k].assign(weights[k]))

# Saves weights from a dictionary of tensorflow variables into an NPZ archive
def _save_weights(variable_dict, weight_file, sess):
    
    keys = sorted(variable_dict.keys())

    # Pray for my RAM usage
    weights = map(lambda k: (k, sess.run(variable_dict[k])), keys)
    np.savez(weight_file, *weights)


class Model:

    def __init__(self, images):
        self.vgg_weights = {}
        self.conv_weights = {}
        self.class_weights = {}
        self.bbox_weights = {}

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = _kernel_variable([3, 3, 3, 64], name='weights', trainable=False)
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _bias_variable([64], name='biases', trainable=False)

            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)

            self.vgg_weights['conv1_1_W'] = kernel
            self.vgg_weights['conv1_1_b'] = biases

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = _kernel_variable([3, 3, 64, 64], name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _bias_variable([64], name='weights', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)

            self.vgg_weights['conv1_2_W'] = kernel
            self.vgg_weights['conv1_2_b'] = biases

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = _kernel_variable([3, 3, 64, 128], name='weights', trainable=False)
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _bias_variable([128], name='weights', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)

            self.vgg_weights['conv2_1_W'] = kernel
            self.vgg_weights['conv2_1_b'] = biases

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = _kernel_variable([3, 3, 128, 128], name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _bias_variable([128], name='weights', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)

            self.vgg_weights['conv2_2_W'] = kernel
            self.vgg_weights['conv2_2_b'] = biases

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = _kernel_variable([3, 3, 128, 256], name='weights', trainable=False)
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _bias_variable([256], name='weights', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)

            self.vgg_weights['conv3_1_W'] = kernel
            self.vgg_weights['conv3_1_b'] = biases

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = _kernel_variable([3, 3, 256, 256], name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _bias_variable([256], name='weights', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)

            self.vgg_weights['conv3_2_W'] = kernel
            self.vgg_weights['conv3_2_b'] = biases

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = _kernel_variable([3, 3, 256, 256], name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _bias_variable([256], name='weights', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)

            self.vgg_weights['conv3_3_W'] = kernel
            self.vgg_weights['conv3_3_b'] = biases
        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = _kernel_variable([3, 3, 256, 512], name='weights', trainable=False)
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _bias_variable([512], name='weights', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)

            self.vgg_weights['conv4_1_W'] = kernel
            self.vgg_weights['conv4_1_b'] = biases

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = _kernel_variable([3, 3, 512, 512], name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _bias_variable([512], name='weights', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)

            self.vgg_weights['conv4_2_W'] = kernel
            self.vgg_weights['conv4_2_b'] = biases

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = _kernel_variable([3, 3, 512, 512], name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _bias_variable([512], name='weights', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)

            self.vgg_weights['conv4_3_W'] = kernel
            self.vgg_weights['conv4_3_b'] = biases

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = _kernel_variable([3, 3, 512, 512], name='weights', trainable=True)
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _bias_variable([512], name='weights', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)

            self.vgg_weights['conv5_1_W'] = kernel
            self.vgg_weights['conv5_1_b'] = biases

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = _kernel_variable([3, 3, 512, 512], name='weights', trainable=True)
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _bias_variable([512], name='weights', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)

            self.vgg_weights['conv5_2_W'] = kernel
            self.vgg_weights['conv5_2_b'] = biases

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = _kernel_variable([3, 3, 512, 512], name='weights', trainable=True)
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _bias_variable([512], name='weights', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)

            self.vgg_weights['conv5_3_W'] = kernel
            self.vgg_weights['conv5_3_b'] = biases

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool5')

        # These are the additional layers added by me to the conv section of the network
        with tf.name_scope('conv6_1') as scope:
            kernel = _kernel_variable([5, 5, 512, 512], name='weights', trainable=True)
            biases = _bias_variable([512], name='weights', trainable=True)

            conv = tf.nn.conv2d(self.pool5, kernel, [1, 2, 2, 1], padding='SAME')

            out = tf.nn.bias_add(conv, biases)
            self.conv6_1 = tf.nn.relu(out, name=scope)

            self.conv_weights['conv6_1_W'] = kernel
            self.conv_weights['conv6_1_b'] = biases

        with tf.name_scope('conv7_1') as scope:
            kernel = _kernel_variable([7, 7, 512, 512], name='weights', trainable=True)
            biases = _bias_variable([512], name='weights', trainable=True)

            conv = tf.nn.conv2d(self.pool5, kernel, [1, 4, 4, 1], padding='SAME')

            out = tf.nn.bias_add(conv, biases)
            self.conv7_1 = tf.nn.relu(out, name=scope)

            self.conv_weights['conv7_1_W'] = kernel
            self.conv_weights['conv7_1_b'] = biases

        self.pool6 = tf.nn.max_pool(self.conv5_3,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 3, 3, 1],
                                    padding='SAME',
                                    name='pool6')

    # Classification head
    def classifier_head(self, num_classes, labels, attach_point=None):
        if not attach_point:
            attach_point = self.pool6

        layer_size = int(np.prod(attach_point.get_shape().as_list()[1:]))
        attach_point_flat = tf.reshape(attach_point, shape=(-1, layer_size))

        with tf.name_scope('fc1') as scope:
            weights = _kernel_variable([layer_size, 2048], name='weights', trainable=True)
            biases = _bias_variable([2048], name='biases', trainable=True)

            fc1 = tf.nn.bias_add(tf.matmul(attach_point_flat, weights), biases)
            self.fc1 = tf.nn.relu(fc1, name=scope)

            self.class_weights['fc1_W'] = weights
            self.class_weights['fc1_b'] = biases

        with tf.name_scope('softmax') as scope:
            weights = _kernel_variable([2048, num_classes], name='weights', trainable=True)
            biases = _bias_variable([num_classes], name='biases', trainable=True)

            fc2 = tf.nn.bias_add(tf.matmul(self.fc1, weights), biases)
            self.softmax = tf.nn.softmax(fc2, name='softmax')

            self.class_weights['softmax_W'] = weights
            self.class_weights['softmax_b'] = biases

        softmax_to_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=fc2)

        return self.softmax, tf.reduce_mean(softmax_to_loss)

    def bbox_head(self, attach_point):
        layer_size = int(np.prod(attach_point.get_shape().as_list()[1:]))
        attach_point_flat = tf.reshape(attach_point, shape=(-1, layer_size))

        with tf.name_scope('bbox_fc1') as scope:
            weights = _kernel_variable([layer_size, 2048], name='weights', trainable=True)
            biases = _bias_variable([2048], name='biases', trainable=True)

            fc1 = tf.nn.bias_add(tf.matmul(attach_point_flat, weights), biases)
            self.bbox_fc1 = tf.nn.relu(fc1, name=scope)

            self.bbox_weights['bbox_fc1_W'] = weights
            self.bbox_weights['bbox_fc1_b'] = biases

        with tf.name_scope('bbox_softmax') as scope:
            weights = _kernel_variable([2048, 4], name='weights', trainable=True)
            biases = _bias_variable([4], name='biases', trainable=True)

            fc2 = tf.nn.bias_add(tf.matmul(self.fc1, weights), biases)
            self.bbox_output = fc2

            self.bbox_weights['bbox_output_W'] = weights
            self.bbox_weights['bbox_output_b'] = biases

        return self.bbox_output

    def create_validator(self, val_input, labels):
        correct = tf.equal(tf.argmax(val_input, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct, tf.float32), name='evaluation')

    def initialize_pre_training(self, sess, classifier=True, bbox_regression=False):
        # Create a list of all variables

        sess.run(tf.variables_initializer(self.conv_weights.values()))

        if classifier:
            sess.run(tf.variables_initializer(self.class_weights.values()))

        if bbox_regression:
            sess.run(tf.variables_initializer(self.bbox_weights.values()))

    def load_vgg_weights(self, weight_file, sess):
        print "Loading pretrained VGG16 weights from %s" % weight_file

        _load_weights(self.vgg_weights, weight_file, sess)

    def save_conv_weights(self, weight_file, sess):
        print "Saving Conv weights at %s" % weight_file
        _save_weights(self.conv_weights, weight_file, sess)

    def load_conv_weights(self, weight_file, sess):
        print "Loading saved Conv weights from %s" % weight_file

        _load_weights(self.conv_weights, weight_file, sess)
        
    def save_class_weights(self, weight_file, sess):
        print "Saving classifier weights at %s" % weight_file

        _save_weights(self.class_weights, weight_file, sess)

    def load_class_weights(self, weight_file, sess):
        print "Loading classifier weights from %s" % weight_file

        _load_weights(self.conv_weights, weight_file, sess)

    def save_bbox_weights(self, weight_file, sess):
        print "Saving bbox regression weights at %s" % weight_file

        _save_weights(self.bbox_weights, weight_file, sess)

    def load_bbox_weights(self, weight_file, sess):
        print "Loading bbox regression weights from %s" % weight_file

        _load_weights(self.bbox_weights, weight_file, sess)


def main(argv):

        images = tf.placeholder(dtype=tf.float32, shape=(None, 210, 280, 3))
        labels = tf.placeholder(dtype=tf.float32, shape=(None, 2))

        vgg = Model(images)

        sess = tf.Session()

        classifier, loss = vgg.classifier_head(num_classes=2, labels=labels)

        training, testing, validation = create_data_sets(data_dir='./data')

        training.set_batch_size(40)
        testing.set_batch_size(40)
        validation.set_batch_size(40)

        validator = vgg.create_validator(classifier, labels)

        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
        sess.run(tf.initialize_all_variables())

        vgg.load_vgg_weights('./data/vgg16_weights.npz', sess)
        vgg.initialize_pre_training(sess, classifier=True, bbox_regression=False)

        epochs_to_train = NUM_EPOCHS
        num_steps_per_epoch = training.get_epoch_steps()
        reporting_step = int(num_steps_per_epoch / 4)

        print "Running %d epochs with %d training steps per epoch" % (epochs_to_train, num_steps_per_epoch)
        for epoch in xrange(epochs_to_train):

            print "Epoch: %d --- Epochs left: %d" % (epoch, epochs_to_train - epoch)
            print "-----------------------------------------------"

            for step in xrange(num_steps_per_epoch):
                sys.stdout.flush()
                sys.stdout.write('\rStep: %d' % (step + 1))

                batch_images, batch_labels = training.next_batch()
                sess.run(train_step, feed_dict={
                    images: batch_images,
                    labels: batch_labels
                })

                if step % reporting_step == 0:
                    loss_val, accuracy = sess.run((loss, validator), feed_dict={
                        images: batch_images,
                        labels: batch_labels
                    })

                    print "\r    Step %d --- loss: %f --- accuracy: %f" % (step, loss_val, accuracy)

            test_images, test_labels = testing.next_batch()
            loss_val, accuracy = sess.run((loss, validator), feed_dict={
                images: test_images,
                labels: test_labels
            })

            print "\nFinal results for epoch %d --- loss: %f --- accuracy: %f" % (epoch, loss_val, accuracy)

        print "-----------------------------------------------"

        print "Loading validation set for final accuracy..."

        final_loss = 0.0
        final_accuracy = 0.0

        for step in xrange(validation.get_epoch_steps()):
            validation_images, validation_labels = validation.next_batch()
            loss_val, accuracy = sess.run((loss, validator), feed_dict={
                images: validation_images,
                labels: validation_labels
            })

            final_loss += loss_val
            final_accuracy += accuracy

        final_loss /= validation.get_epoch_steps()
        final_accuracy /= validation.get_epoch_steps()

        print "Validation data results: --- loss: %f --- accuracy: %f" % (final_loss, final_accuracy)

        vgg.save_conv_weights('./data/conv_weights.npz', sess)
        vgg.save_class_weights('./data/class_weights.npz', sess)

if __name__ == "__main__":
    main(sys.argv[1:])
