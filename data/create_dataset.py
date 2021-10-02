from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from .clean_data import Dataset

import tensorflow as tf
import numpy as np


def data_limiter(num, total_captions, all_img_name_vector):
    train_captions, img_name_vector = shuffle(total_captions, all_img_name_vector, random_state=1)
    train_captions = train_captions[:num]
    img_name_vector = img_name_vector[:num]
    return train_captions, img_name_vector


def load_pre_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (64, 64))
    return img


def tokenize(all_captions, all_img_name_vector):
    train_captions, img_name_vector = data_limiter(40000, all_captions, all_img_name_vector)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, oov_token='<unk>',
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)

    train_seqs = tokenizer.texts_to_sequences(train_captions)
    tokenizer.word_index['<pad>'] = 0
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector, cap_vector, test_size=0.2,
                                                                        random_state=0)

    return tokenizer, (img_name_train, img_name_val, cap_train, cap_val)


def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap, img_name


def create_data_tensor(img_name, cap_name, batch_siz=64, shuffle=128):
    dataset = tf.data.Dataset.from_tensor_slices((img_name, cap_name))
    dataset = dataset.map(
        lambda item1, item2: tf.numpy_function(
            map_func, [item1, item2], [tf.float32, tf.int32, tf.string, tf.float32]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.shuffle(shuffle).batch(batch_siz)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


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
    # tokenize above data.
    tokenizer, (img_name_train, img_name_val, cap_train, cap_val) = tokenize(all_captions, all_img_name_vector)
    # converting data into train and test set tensors.
    dataset = create_data_tensor(img_name_train, cap_train, batch_siz=64)
    i_data = create_data_tensor(img_name_val, cap_val, batch_siz=1)

    return dataset, i_data, tokenizer
