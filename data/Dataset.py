from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from .ProcessData import Dataset

from evaluation import Cider
import tensorflow as tf
import numpy as np


def data_limiter(num, captions, img_name_vector):
    t_cap = captions[:num]
    i_name_vec = img_name_vector[:num]
    t_cap, i_name_vec = shuffle(t_cap, i_name_vec, random_state=1)
    return t_cap, i_name_vec


def tokenize(all_captions, all_img_name_vector, data_limit=40000):
    train_captions, img_name_vector = data_limiter(data_limit, all_captions, all_img_name_vector)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, oov_token='<unk>',
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    tokenizer.fit_on_texts(train_captions)

    train_seqs = tokenizer.texts_to_sequences(train_captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector, cap_vector, test_size=0.1,
                                                                        random_state=0)

    return tokenizer, (img_name_train, img_name_val, cap_train, cap_val)


def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    # full_img = np.load(full_img.decode('utf-8') + '.npy')
    return img_tensor, cap, img_name


def create_data_tensor(img_name, cap_name, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices((img_name, cap_name))
    dataset = dataset.map(
        lambda item1, item2: tf.numpy_function(
            map_func, [item1, item2], [tf.float32, tf.int32, tf.string]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.shuffle(128).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def full_feature_path(path, img_name):
    full_img_path = []
    for i in img_name:
        *_, img = i.split('/')
        full_img_path.append(path + img)

    return full_img_path


def create_dataset(cfg):
    paths = {
        "image_path": cfg['IMG_PATH'],
        "text_path": cfg["TXT_PATH"],
        "cap_file": cfg["CAP_FILE"],
        "img_name": cfg["IMG_NAME"],
        "dataset": cfg["DATASET_NAME"]
    }

    # get mapping between images -> captions.
    all_captions, all_img_name_vector = Dataset(**paths)

    # creating cider
    _tmp = dict(zip(all_img_name_vector, all_captions))
    cider = Cider(_tmp)

    # tokenize above data.
    tokenizer, (img_name_train, img_name_val, cap_train, cap_val) = tokenize(all_captions, all_img_name_vector,
                                                                             cfg['DATASET_SIZE'])

    # full_img_name_train = full_feature_path(cfg['F_IMG_PATH'], img_name_train)
    # full_img_name_val = full_feature_path(cfg['F_IMG_PATH'], img_name_val)

    # converting data into train and test set tensors.
    dataset = create_data_tensor(img_name_train, cap_train, batch_size=cfg['BATCH_SIZE'])
    i_data = create_data_tensor(img_name_val, cap_val, batch_size=1)

    return dataset, i_data, tokenizer, cider
