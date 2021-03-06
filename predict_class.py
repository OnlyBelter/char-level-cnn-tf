#! /usr/bin/env python
# based on ideas in https://github.com/dennybritz/cnn-text-classification-tf

import tensorflow as tf
import numpy as np
import os
import time
from datetime import datetime
try:
    import preprocessing
    from model import CharCNN
except ImportError:
    from .model import CharCNN
# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("n_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
print('this is epochs', FLAGS.n_epochs)
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
data_dir = r'data'
# can download yelp dataset from below url
# https://raw.githubusercontent.com/intuinno/YelpVis/master/0DataSet/yelp_academic_dataset_review.json
training_data_path = os.path.join(data_dir, 'train.txt')
dev_data_path = os.path.join(data_dir, 'dev.txt')
print("Loading data...")
x, y = preprocessing.load_data(data_path=dev_data_path)
# Randomly shuffle data
np.random.seed(10)
# shuffle_indices = np.random.permutation(np.arange(len(y)))
# x_shuffled = x[shuffle_indices]
# y_shuffled = y[shuffle_indices]
# Split train/test set
# n_dev_samples = int(len(y) * 0.3)
# TODO: Create a fuckin' correct cross validation procedure
x_dev = x
y_dev = y
print("Dev data sample number: {:d}".format(len(y_dev)))



# Training
# ==================================================

with tf.Graph().as_default():
    ## define training computation graph
    learning_rate = 0.01
    m, n = x_dev.shape
    print('x_dev\'s shape is', x_dev.shape)
    print(x_dev[0])
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    cnn = CharCNN()

    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    train_op = optimizer.minimize(cnn.loss)
    # init = tf.global_variables_initializer()
    n_batches = int(np.ceil(m / FLAGS.batch_size))

    # create a Saver node after all variable nodes are created
    saver = tf.train.Saver()

    # Output directory for models and summaries
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", now))
    print("Writing to {}\n".format(out_dir))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "cv"))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    root_logdir = 'tf_logs'
    logdir = '{}/run-{}'.format(root_logdir, now)

    # summary node can used by TensorBoard
    loss_summary = tf.summary.scalar('LOSS', cnn.loss)
    file_writer = tf.summary.FileWriter(logdir=logdir, graph=tf.get_default_graph())
    current_loss = 10

    with sess.as_default():
        # load saved model
        saver.restore(sess, 'cv/my_model_current_best.ckpt')
        # predict by saved model...
        X_batch, y_batch = preprocessing.fetch_batch_for_predict(
            x_dev, y_dev, batch_size=1000)
        # print(y_batch)
        feed_dict = {
            cnn.input_x: np.float32(X_batch),
            cnn.input_y: np.float32(y_batch),
            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
        }
        # no need to train
        loss, accuracy, input_y_class, y_pred, scores = sess.run([cnn.loss, cnn.accuracy,
                                      tf.argmax(cnn.input_y, 1), cnn.predictions, cnn.scores], feed_dict)
        print('MSE =', loss, 'Accuracy =', accuracy,
              'y = ', input_y_class, 'y_pred = ', y_pred)
