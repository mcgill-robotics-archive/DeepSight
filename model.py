
import tensorflow as tf
from input_pipeline import inputs
import ops
from os import path, listdir
from sys import stdout

THIRTYMIN_COLLECTION = 'thirty_min_collection'

if __name__ == "__main__":
    flags = tf.app.flags
    flags.DEFINE_string('data_dir', 'data', 'The directory containing the data')
    flags.DEFINE_integer('batch_size', 10, 'Batch size to train with')
    flags.DEFINE_integer('max_steps', 3000, 'Number of steps to train for')
    flags.DEFINE_float('learning_rate', 1e-3, 'The learning rate')
    flags.DEFINE_float('adam_beta1', 0.9, 'Adam hyper parameter beta1')
    flags.DEFINE_float('adam_beta2', 0.999, 'Adam hyper parameter beta2')
    flags.DEFINE_float('adam_epsilon', 1e-8, 'Adam hyper parameter epsilon')

    flags.DEFINE_float('class_weight', 1.0, 'Weight for classifier loss')
    flags.DEFINE_float('bbox_weight', 1e2, 'Weight for bbox loss')

    flags.DEFINE_integer('report_step', 50, 'The training step to report on')
    flags.DEFINE_integer('save_step', 500, 'The training step to save on')
    flags.DEFINE_integer('max_test_steps', 500, 'The maximum number of test steps')

    flags.DEFINE_string('logdir', 'data/logdir', 'Where the save model and summaries')
    flags.DEFINE_string('model_name', 'thirty_min', 'The name of the model')

    FLAGS = flags.FLAGS


def create_model(images, trainable=False, reuse=False):

    with tf.variable_scope('conv_1', reuse=reuse):
        model = ops.conv_layer(images, filter_size=(5, 5), num_features=32, strides=(1, 1),
                               trainable=trainable, collection=THIRTYMIN_COLLECTION)
        model = ops.instance_norm(model)
        model = ops.max_pool(model, pool_size=(3, 3), strides=(2, 2))

    with tf.variable_scope('conv_2', reuse=reuse):
        model = ops.conv_layer(model, filter_size=(5, 5), num_features=64, strides=(1, 1),
                               trainable=trainable, collection=THIRTYMIN_COLLECTION)
        model = ops.instance_norm(model)
        model = ops.max_pool(model, pool_size=(7, 7), strides=(4, 4))

    with tf.variable_scope('flatten'):
        model_shape = model.get_shape().as_list()
        layer_size = reduce(lambda x, y: x * y, model_shape[1:], 1)
        model = tf.reshape(model, shape=(-1, layer_size), name='reshape')

    # Classifier
    with tf.variable_scope('class_fc_1', reuse=reuse):
        classifier = tf.contrib.layers.fully_connected(model, 1024, activation_fn=tf.nn.relu)

    with tf.variable_scope('class_softmax', reuse=reuse):
        classifier_logits = tf.contrib.layers.fully_connected(classifier, 2, activation_fn=None)
        classifier = tf.nn.softmax(classifier_logits)


    # BBox head
    with tf.variable_scope('bbox_fc_1', reuse=reuse):
        bbox = tf.contrib.layers.fully_connected(model, 1024, activation_fn=tf.nn.relu)

    with tf.variable_scope('bbox_output', reuse=reuse):
        bbox = tf.contrib.layers.fully_connected(bbox, 4, activation_fn=tf.nn.sigmoid)

    return classifier, bbox, classifier_logits


def model_variables():
    return tf.get_collection(THIRTYMIN_COLLECTION)


def initializer():
    return tf.variables_initializer(tf.get_collection(THIRTYMIN_COLLECTION))


def model_validate(class_scores, bbox_head, class_labels, bbox_labels):

    with tf.name_scope('class_accuracy'):
        network_predictions = tf.argmax(class_scores, axis=1)
        label_predictions = tf.argmax(class_labels, axis=1)
        class_accuracy = tf.reduce_mean(tf.cast(tf.equal(network_predictions, label_predictions), dtype=tf.float32), name='accuracy')

    with tf.name_scope('bbox_accuracy'):
        bbox_accuracy = tf.sqrt(tf.reduce_mean((bbox_head - bbox_labels) ** 2), name='accuracy')

    return class_accuracy, bbox_accuracy

def model_loss(class_logits, bbox_head, class_labels, bbox_labels, summaries=True, summary_scope='loss_summaries'):

    with tf.name_scope('class_loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(class_logits, class_labels)
        class_loss = tf.mul(tf.reduce_mean(cross_entropy), FLAGS.class_weight, name='class_loss')

    with tf.name_scope('bbox_loss'):
        mse = tf.sqrt(tf.reduce_mean((bbox_head - bbox_labels) ** 2), name='mse')
        bbox_loss = tf.mul(mse, FLAGS.bbox_weight, name='bbox_loss')

    total_loss = tf.add(class_loss, bbox_loss, name='total_loss')

    if summaries:
        with tf.name_scope(summary_scope):
            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('bbox_loss', bbox_loss)
            tf.summary.scalar('total_loss', total_loss)

    return total_loss


def main(argv):

    image_dir = path.join(FLAGS.data_dir, 'img')
    image_names = [path.basename(f).split('.')[0] for f in listdir(image_dir) if f.endswith('.jpg')]

    # Filter images without labels
    image_names = filter(lambda name: path.exists(path.join(FLAGS.data_dir, 'csv_labels', name+'.csv')), image_names)

    num_examples = len(image_names)
    training_reserve = 0.7

    training = image_names[:int(training_reserve * float(num_examples))]
    testing = image_names[int(training_reserve * float(num_examples)):]

    image_batch, label_batch = inputs(FLAGS.data_dir, training, image_size=(210, 280), batch_size=FLAGS.batch_size, num_threads=2)

    class_labels, bbox_labels, _, _ = label_batch

    with tf.variable_scope('thirty_min_net', reuse=False):
        classifier, bbox_head, class_logits = create_model(image_batch, trainable=True, reuse=False)

    test_image_batch, test_label_batch = inputs(FLAGS.data_dir, testing, image_size=(210, 280), batch_size=FLAGS.batch_size, num_threads=1)

    test_class_labels, test_bbox_labels, _, _ = test_label_batch

    with tf.variable_scope('thirty_min_net', reuse=True):
        test_classifier, test_bbox_head, _ = create_model(test_image_batch, trainable=False, reuse=True)

    with tf.name_scope('train_validator'):
        train_validator = model_validate(classifier, bbox_head, class_labels, bbox_labels)

    with tf.name_scope('test_validator'):
        test_validator = model_validate(test_classifier, test_bbox_head, test_class_labels, test_bbox_labels)

    loss = model_loss(class_logits, bbox_head, class_labels, bbox_labels)

    global_step = tf.Variable(0, dtype=tf.int32, name='global_step', collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
                                       beta1=FLAGS.adam_beta1,
                                       beta2=FLAGS.adam_beta2,
                                       epsilon=FLAGS.adam_epsilon)
    train_step = optimizer.minimize(loss, global_step=global_step)

    print "Merging summaries"
    with tf.name_scope('weight_summaries'):
        for var in model_variables():
            ops.variable_summaries(var)

    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logdir=FLAGS.logdir)

    sess = tf.Session()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.Saver([global_step] + model_variables())

    print "Initializing variables..."
    sess.run(tf.group(
        tf.global_variables_initializer(),
        initializer(),
        tf.local_variables_initializer()
    ))

    while not coord.should_stop():

        _, summary = sess.run([train_step, merged_summary])

        step = sess.run(global_step)

        summary_writer.add_summary(summary, step)

        stdout.flush()
        stdout.write('\rstep: %i' % step)
        if (step - 1) % FLAGS.report_step == 0:
            loss_val, class_acc, train_acc = sess.run([loss] + list(train_validator))
            print "\r Summary for step %i --- Loss: %f --- Class Accuracy: %f --- BBox MSE: %f" % (step, loss_val, class_acc, train_acc)
        if (step - 1) % FLAGS.save_step == 0:
            saver.save(sess, path.join(FLAGS.logdir, FLAGS.model_name + '.ckpt'), global_step=step, write_meta_graph=False)
        if (step - 1) >= FLAGS.max_steps:
            break

    print "\rSaving final model"
    step = sess.run(global_step)
    saver.save(sess, path.join(FLAGS.logdir, FLAGS.model_name + '.ckpt'), global_step=step, write_meta_graph=True)

    print "Running validation on test set"
    class_acc, bbox_acc = 0.0, 0.0
    num_test_steps = min(int((1 - training_reserve) * num_examples / FLAGS.batch_size),
                         FLAGS.max_test_steps)
    for test_step in xrange(num_test_steps):
        stdout.flush()
        test_class_acc, test_bbox_acc = sess.run(list(test_validator))
        stdout.write('\rTest step: %i --- Test Class Accuracy: %f --- Test BBox MSE: %f' % (test_step, test_class_acc, test_bbox_acc))

        class_acc += test_class_acc
        bbox_acc += test_bbox_acc

    class_acc /= num_test_steps
    bbox_acc /= num_test_steps

    coord.request_stop()
    coord.join(threads)

    print "\rAverage Test Accuracies --- Class: %f --- BBox: %f" % (class_acc, bbox_acc)


if __name__ == "__main__":
    tf.app.run(main)