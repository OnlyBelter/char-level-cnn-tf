# based on ideas from https://github.com/dennybritz/cnn-text-classification-tf

import numpy as np
import json
import unicodedata
import os


def load_yelp(alphabet, data_path):
    examples = []
    labels = []
    with open(data_path) as f:
        i = 0
        for line in f:
            review = json.loads(line)
            # print(review)
            stars = review["stars"]
            text = review["text"]
            # if stars != 3:
            text_end_extracted = extract_end(list(text.lower()))
            padded = pad_sentence(text_end_extracted)
            text_int8_repr = string_to_int8_conversion(padded, alphabet)
            if stars <= 5:
                labels.append(stars)
                examples.append(text_int8_repr)
            i += 1
            if i % 10000 == 0:
                print("Non-neutral instances processed: " + str(i))
    labels_vec = convert_y_to_logic_vector(labels, class_num=5, start_inx=0)
    return examples, labels_vec


def load_movie_comment(alphabet, data_path):
    examples = []
    labels = []
    with open(data_path) as f:
        i = 0
        for line in f:
            each_line = line.strip()
            # print(each_line)
            stars = int(each_line[0])
            text = each_line[1:].strip().lower()
            # if stars != 3:
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
            text_end_extracted = extract_end(list(text.lower()))
            padded = pad_sentence(text_end_extracted)
            text_int8_repr = string_to_int8_conversion(padded, alphabet)
            if stars <= 5:
                labels.append(stars)
                examples.append(text_int8_repr)
            i += 1
            if i % 2000 == 0:
                print("Non-neutral instances processed: " + str(i))
    labels_vec = convert_y_to_logic_vector(labels, class_num=5, start_inx=0)
    return examples, labels_vec


def convert_y_to_logic_vector(y_list, class_num, start_inx=1):
    """
    convert 4 to [0, 0, 0, 1]
    :param y_list:
    :return:
    """
    new_y = []
    for i in y_list:
        i_vec = [0] * class_num
        i_vec[i-start_inx] = 1
        new_y.append(i_vec)
    return new_y

# a = [1, 4, 3, 1, 4, 1, 2, 4, 3, 2]
# print(convert_y_to_logic_vector(a, class_num=4))

def extract_end(char_seq):
    if len(char_seq) > 1014:
        char_seq = char_seq[-1014:]
    return char_seq


def pad_sentence(char_seq, padding_char=" "):
    char_seq_length = 1014
    num_padding = char_seq_length - len(char_seq)
    new_char_seq = char_seq + [padding_char] * num_padding
    return new_char_seq


def string_to_int8_conversion(char_seq, alphabet):
    x = np.array([alphabet.find(char) for char in char_seq], dtype=np.int8)
    return x


def get_batched_one_hot(char_seqs, labels):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
    x_batch = char_seqs[:]
    y_batch = labels[:]
    x_batch_one_hot = np.zeros(shape=[len(x_batch), len(alphabet), len(x_batch[0]), 1])
    for example_i, char_seq_indices in enumerate(x_batch):
        for char_pos_in_seq, char_seq_char_ind in enumerate(char_seq_indices):
            if char_seq_char_ind != -1:
                x_batch_one_hot[example_i][char_seq_char_ind][char_pos_in_seq][0] = 1
    return [x_batch_one_hot, y_batch]


def load_data(data_path):
    # TODO Add the new line character later for the yelp'cause it's a multi-line review
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
    examples, labels = load_movie_comment(alphabet, data_path)
    x = np.array(examples, dtype=np.int8)
    y = np.array(labels, dtype=np.int8)
    print("x_char_seq_ind=" + str(x.shape))
    print("y shape=" + str(y.shape))
    return [x, y]


def fetch_batch(X, y, batch_size=128, class_num=5):
    """
    Generates a batch, and each class has equal samples
    """
    n_each_class = int(np.ceil(batch_size/class_num))
    n_each_class_ = np.random.randint(n_each_class-10, n_each_class+10, 5)
    y_tag = y.argmax(axis=1)
    class2inx = {}  # class to indices
    for i in range(class_num):
        inx = np.where(y_tag==i)
        np.random.shuffle(inx[0])
        class2inx[i] = list(inx[0])
    indices = []
    for i in class2inx:
        indices += class2inx[i][:n_each_class_[i]]
    np.random.shuffle(indices)
    X_batch = X[indices]
    y_batch = y[indices]
    return get_batched_one_hot(X_batch, y_batch)


def fetch_batch_for_predict(X, y, batch_size=-1):
    """
    process data for predict
    """
    X_batch = X[:batch_size]
    y_batch = y[:batch_size]
    return get_batched_one_hot(X_batch, y_batch)
