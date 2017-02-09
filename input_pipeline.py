
from os import path, listdir
from load_data import load_label
import tensorflow as tf

if __name__ == "__main__":
    flags = tf.app.flags
    flags.DEFINE_string('label_dir', 'data/label', 'The directory with all the labels')
    flags.DEFINE_string('outdir', 'data/csv_labels', 'The directory to output the csv labels')
    FLAGS = flags.FLAGS

def create_line(label):

    # [ [[x, y, w, h], 1.0] ]
    if len(label) == 0:
        list = [1.0, 0.0] + [-1.0] * 12
    else:
        list = [0.0, 1.0] + [-1.0] * 12
        for i in xrange(min(len(label), 3)):
            # Write the bounding box
            list[(2+4*i):(6+4*i)] = label[i][0]

    literals = map(lambda f: '%f' % f, list)
    return ','.join(literals)


def create_labels(label_dir, output_dir):

    label_names = [f.split('.')[0] for f in listdir(label_dir) if f.endswith('.txt')]
    label_files = [path.join(label_dir, f + '.txt') for f in label_names]

    labels = map(load_label, label_files)

    csv_lines = map(create_line, labels)

    for name, line in zip(label_names, csv_lines):
        outfile = path.join(output_dir, '%s.csv' % name)
        with open(outfile, 'w') as file_stream:
            file_stream.write(line)


def decode_jpeg(queue, image_size):
    reader = tf.WholeFileReader(name='reader')
    _, record = reader.read(queue, name='read')
    image = tf.image.decode_jpeg(record, channels=3, name='raw_image')
    image = tf.cast(tf.reshape(image, shape=list(image_size) + [3], name='reshape'), dtype=tf.float32, name='image')

    return image


def decode_label(queue):
    reader = tf.WholeFileReader(name='reader')
    _, record = reader.read(queue, name='read')
    record_defaults = [[0.0], [1.0]] + [[-1.0]] * 12
    csv_body = tf.decode_csv(record, record_defaults=record_defaults, name='raw_label')
    class_label = tf.pack(csv_body[0:2], name='class_label')
    bbox1 = tf.pack(csv_body[2:6], name='bbox_1')
    bbox2 = tf.pack(csv_body[6:10], name='bbox_2')
    bbox3 = tf.pack(csv_body[10:14], name='bbox_3')

    return class_label, bbox1, bbox2, bbox3

def inputs(data_dir, name_list, image_size, batch_size, num_threads=2):

    image_files = [path.join(data_dir, 'img', name + '.jpg') for name in name_list]
    label_files = [path.join(data_dir, 'csv_labels', name + '.csv') for name in name_list]

    with tf.name_scope('input'):
        image_files = tf.constant(image_files, dtype=tf.string, name='image_files')
        label_files = tf.constant(label_files, dtype=tf.string, name='label_files')

        with tf.name_scope('producers'):
            image_producer = tf.train.string_input_producer(image_files, shuffle=True, capacity=32, name='images')
            label_producer = tf.train.string_input_producer(label_files, shuffle=True, capacity=32, name='labels')

        with tf.name_scope('image_input'):
            image = decode_jpeg(image_producer, image_size)

        with tf.name_scope('label_input'):
            label = decode_label(label_producer)

        min_after_dequeue = 32
        capacity = 32 + 3 * batch_size
        # Label is a tuple containing (class_label, bbox1, bbox2, bbox3)
        batch_tensors = tf.train.shuffle_batch([image] + list(label), batch_size=batch_size,
                                                          capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue,
                                                          num_threads=num_threads,
                                                          name='batch_producer')
        image_batch = batch_tensors[0]
        label_batch = batch_tensors[1:]

    # Label batch is also a tuple with (class_label_batch, bbox1_batch, bbox2_batch, bbox3_batch)
    return image_batch, label_batch


def main(argv):
    create_labels(FLAGS.label_dir, FLAGS.outdir)

if __name__ == "__main__":
    tf.app.run(main)