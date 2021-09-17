import os
import string
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
# import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers

image_path = './Dataset/Flicker8k_Dataset/'
dir_Flickr_text = './Dataset/Flickr8k.token.txt'

jpgs = os.listdir(image_path)
print(f'Total image in dataset is {len(jpgs)}')

file = open(dir_Flickr_text, 'r')
text = file.read()
file.close()

datapxt = []
for line in text.split('\n'):
    col = line.split('\t')
    if len(col) == 1:
        continue

    w = col[0].split('#')
    t = (w[0], col[1].lower())
    datapxt.append(t)

npic = 5
npix = 224
target_size = (npix, npix, 3)
count = 1

vocabulary = []
for i in datapxt:
    vocabulary.extend(i[1].split())

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


datafinalpxt = []
for i in datapxt:
    #     print(i[1])
    newcaption = text_clean(i[1])
    t = (i[0], newcaption)
    datafinalpxt.append(t)

clean_vocab = []

pt = pd.DataFrame(datafinalpxt, columns=['filename', 'captions'])
import pickle

pickle.dump(pt, open('tuple.dump', 'wb'))
data = pickle.load(open('tuple.dump', 'rb'))

data = data[data.filename != '2258277193_586949ec62.jpg.1']
all_captions = []

for caption in data['captions'].astype(str):
    caption = '<start> ' + caption + ' <end>'
    all_captions.append(caption)

all_img_name_vector = []
for annot in data['filename']:
    full_image_path = image_path + annot
    all_img_name_vector.append(full_image_path)


# def data_limiter(num, total_captions, all_img_name_vector):
#     train_captions, img_name_vector = shuffle(total_captions, all_img_name_vector, random_state=1)
#     train_captions = train_captions[:num]
#     img_name_vector = img_name_vector[:num]
#     return train_captions, img_name_vector
#
#
# train_captions, img_name_vector = data_limiter(40000, all_captions, all_img_name_vector)


def load_image(image_path):
    print(image_path)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, (299, 299))
    img_inception = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.cast(img, tf.uint8)

    gray = tf.image.rgb_to_grayscale(
        img, name=None
    )
    gray = tf.stack([gray] * 3, axis=-1)
    gray = tf.reshape(
        gray, (299, 299, 3), name=None
    )

    return img, image_path, img_inception, gray


input_shape = (299, 299, 3)
classes = 10
inputs = keras.Input(shape=input_shape)
# Augment images
flip_horizontal = layers.experimental.preprocessing.RandomFlip("horizontal")(inputs)
rnadom_contrast = layers.experimental.preprocessing.RandomContrast(factor=0.5)(inputs)

flip_horizontal = layers.experimental.preprocessing.Rescaling(1.0 / 255)(flip_horizontal)
rnadom_contrast = layers.experimental.preprocessing.Rescaling(1.0 / 255)(rnadom_contrast)
# Add the rest of the model
outputs_flip_horizontal = flip_horizontal
outputs_rnadom_contrast = rnadom_contrast
model = keras.Model(inputs, [outputs_flip_horizontal, outputs_rnadom_contrast])

image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

encode_train = sorted(set(all_img_name_vector))
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(64)

print('hello')
x = 0
for idx, (img, path, img_inception, gray) in enumerate(image_dataset):
    batch_features = model(img)
    batch_features_inception = image_features_extract_model(img_inception)
    batch_features_inception = tf.reshape(batch_features_inception,
                                          (batch_features_inception.shape[0], -1, batch_features_inception.shape[3]))

    batch_features_FH = image_features_extract_model(batch_features[0])
    batch_features_FH = tf.reshape(batch_features_FH,
                                   (batch_features_FH.shape[0], -1, batch_features_FH.shape[3]))

    batch_features_CN = image_features_extract_model(batch_features[1])
    batch_features_CN = tf.reshape(batch_features_CN,
                                   (batch_features_CN.shape[0], -1, batch_features_CN.shape[3]))

    batch_features_gray = image_features_extract_model(gray)
    batch_features_gray = tf.reshape(batch_features_gray,
                                     (batch_features_gray.shape[0], -1, batch_features_gray.shape[3]))
    print(f'batch: {idx}')
    for FH, CN, p, inception, gray_features in zip(batch_features_FH, batch_features_CN, path, batch_features_inception,
                                                   batch_features_gray):
        a = tf.stack([FH, CN, inception, gray_features])
        x = x + 1

        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, a.numpy())
