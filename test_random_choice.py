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
from sklearn.preprocessing import StandardScaler  #  用于数据缩放

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("n_epochs", 1000, "Number of training epochs (default: 200)")
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
X, y = preprocessing.load_data(data_path=training_data_path)

# scaler = StandardScaler()
# scaled_X = scaler.fit_transform(X)

# Split train/test set
# n_dev_samples = int(len(y) * 0.3)
# TODO: Create a fuckin' correct cross validation procedure
print('the first line of X', X[0])
x_train = X
y_train = y
print("Train data sample number: {:d}".format(len(y_train)))


def fetch_batch(X, y, batch_size=200, class_num=5):
    """
    Generates a batch, and each class has equal samples
    """
    n_each_class = int(np.ceil(batch_size/class_num))
    y_tag = y.argmax(axis=1)
    class2inx = {}  # class to indices
    for i in range(class_num):
        inx = np.where(y_tag==i)
        np.random.shuffle(inx[0])
        class2inx[i] = list(inx[0])
    indices = []
    for i in class2inx:
        indices += class2inx[i][:n_each_class]
    np.random.shuffle(indices)
    X_batch = X[indices]
    y_batch = y[indices]
    return (X_batch, y_batch)

print(fetch_batch(x_train, y_train))
