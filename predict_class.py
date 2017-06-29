import tensorflow as tf
import numpy as np
import os
import time
import datetime
try:
    import preprocessing
    from model import CharCNN
except ImportError:
    from .model import CharCNN


# Load data
print("Loading data...")
x, y = preprocessing.load_data()
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
# Split train/test set
n_dev_samples = int(len(y) * 0.3)
# TODO: Create a fuckin' correct cross validation procedure
x_train, x_dev = x_shuffled[:-n_dev_samples], x_shuffled[-n_dev_samples:]
y_train, y_dev = y_shuffled[:-n_dev_samples], y_shuffled[-n_dev_samples:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


cnn = CharCNN()
global_step = tf.Variable(0, name="global_step", trainable=False)
# Dev summaries
# dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
# dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
# dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)


def dev_step(x_batch, y_batch, writer=None):
    """
    Evaluates model on a dev set
    """
    dev_size = len(x_batch)
    max_batch_size = 500
    num_batches = int( dev_size /max_batch_size)
    acc = []
    losses = []
    print("Number of batches in dev set is " + str(num_batches))
    for i in range(num_batches):
        x_batch_dev, y_batch_dev = preprocessing.get_batched_one_hot(
            x_batch, y_batch, i * max_batch_size, (i + 1) * max_batch_size)
        feed_dict = {
            cnn.input_x: x_batch_dev,
            cnn.input_y: y_batch_dev,
            cnn.dropout_keep_prob: 1.0
        }
        with tf.Session() as sess:
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            acc.append(accuracy)
            losses.append(loss)
        time_str = datetime.datetime.now().isoformat()
        print("batch " + str(i + 1) + " in dev >>" +
              " {}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))
        if writer:
            writer.add_summary(summaries, step)
    print("\nMean accuracy=" + str(sum(acc ) /len(acc)))
    print("Mean loss=" + str(sum(losses ) /len(losses)))


if current_step % FLAGS.evaluate_every == 0:
    print("\nEvaluation:")
    dev_step(x_dev, y_dev, writer=dev_summary_writer)
    print("")