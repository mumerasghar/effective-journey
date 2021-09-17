import os
import pickle5 as pickle
import string
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

npic = 5
npix = 224
target_size = (npix, npix, 3)
count = 1
num_layer = 4
BATCH_SIZE = 16
BUFFER_SIZE = 500
top_k = 5000
# target_vocab_size = top_k + 1
target_vocab_size = 5000 + 1
image_path = './Dataset/Flicker8k_Dataset/'
dir_Flickr_text = './Dataset/Flickr8k.token.txt'

data: pd.DataFrame


def do_preprocess():
    global data
    jpgs = os.listdir(image_path)
    print(f'Total image in dataset is {len(jpgs)}')

    file = open(dir_Flickr_text, 'r')
    text = file.read()
    file.close()

    datatxt = []
    for line in text.split('\n'):
        col = line.split('\t')
        if len(col) == 1:
            continue

        w = col[0].split('#')
        datatxt.append(w + [col[1].lower()])

    data = pd.DataFrame(datatxt, columns=['filename', 'index', 'captions'])
    data = data.reindex(columns=['index', 'filename', 'captions'])
    data = data[data.filename != '2258277193_586949ec62.jpg.1']
    uni_filenames = np.unique(data.filename.values)

    vocabulary = []
    for txt in data.captions.values:
        vocabulary.extend(txt.split())

    print(f'Vocublary Size {len(set(vocabulary))}')

    def remove_punctutation(text_original):
        text_no_punctuation = text_original.translate(str.maketrans('', '', string.punctuation))
        return (text_no_punctuation)

    def remove_single_character(text):
        text_len_more_than1 = ''
        for word in text.split():
            if len(word) > 1:
                text_len_more_than1 += ' ' + word

        return (text_len_more_than1)

    def remove_numeric(text):
        text_no_numeric = ''

        for word in text.split():
            isalpha = word.isalpha()

            if isalpha:
                text_no_numeric += ' ' + word

        return (text_no_numeric)

    def text_clean(text_original):
        text = remove_punctutation(text_original)
        text = remove_single_character(text)
        text = remove_numeric(text)

        return (text)

    for i, caption in enumerate(data.captions.values):
        newcaption = text_clean(caption)
        data['captions'].iloc[1] = newcaption

    clean_vocab = []

    for txt in data.captions.values:
        clean_vocab.extend(txt.split())

    print(len(set(clean_vocab)))


all_captions = []
cap_file = './Dataset/captions.pickle'

# check if all_captions is already available
if os.path.isfile(cap_file):
    print("found cached caption.pickle")
    with open(cap_file, 'rb') as f:
        all_captions = pickle.load(f)
else:
    with open(cap_file, 'wb') as f:
        do_preprocess()
        for caption in data['captions'].astype(str):
            caption = '<start> ' + caption + ' <end>'
            all_captions.append(caption)

        pickle.dump(all_captions, f, protocol=pickle.HIGHEST_PROTOCOL)

# check if all_img_name_vector is already available
all_img_name_vector = []
img_name = './Dataset/img_name.pickle'

if os.path.isfile(img_name):
    print('found cached img_name.pickle')
    with open(img_name, 'rb') as f:
        all_img_name_vector = pickle.load(f)
else:
    with open(img_name, 'wb') as f:
        for ann in data['filename']:
            full_image_path = image_path + ann
            all_img_name_vector.append(full_image_path)

        pickle.dump(all_img_name_vector, f, protocol=pickle.HIGHEST_PROTOCOL)


def data_limiter(num, total_captions, all_img_name_vector):
    train_captions, img_name_vector = shuffle(total_captions, all_img_name_vector, random_state=1)
    train_captions = train_captions[:num]
    img_name_vector = img_name_vector[:num]
    return train_captions, img_name_vector


train_captions, img_name_vector = data_limiter(40000, all_captions, all_img_name_vector)


def load_image(image_path):
    print(image_path)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


#
#
# image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
# new_input = image_model.input
# hidden_layer = image_model.layers[-1].output
# image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
#
# encode_train = sorted(set(img_name_vector))
# image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
# image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(64)
#
# for img, path in image_dataset:
#     batch_features = image_features_extract_model(img)
#     batch_features = tf.reshape(batch_features,
#                                 (batch_features.shape[0], -1, batch_features.shape[3]))
#
#     for bf, p in zip(batch_features, path):
#         path_of_feature = p.numpy().decode("utf-8")
#         np.save(path_of_feature, bf.numpy())

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, oov_token='<unk>',
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)

train_seqs = tokenizer.texts_to_sequences(train_captions)
tokenizer.word_index['<pad>'] = 0
train_seqs = tokenizer.texts_to_sequences(train_captions)
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector, cap_vector, test_size=0.2,
                                                                    random_state=0)

num_steps = len(img_name_train)


def load_pre_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (64, 64))
    return img


def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    image = load_pre_image(img_name.decode('utf-8'))
    return img_tensor, cap, img_name, image


dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
dataset = dataset.map(
    lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32, tf.string, tf.float32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
