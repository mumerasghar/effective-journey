import os
import numpy as np
from tensorflow import keras
import tensorflow as tf


def dir_files(file_pth):
    _files = os.listdir(file_pth)
    img_vec = [i for i in _files if not i.endswith('.npy')]
    img_vec = [file_pth + i for i in img_vec]

    return img_vec


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def inception_features(file_pth, BATCH_SIZE=128):
    all_img_name_vector = dir_files(file_pth)
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    encode_train = sorted(set(all_img_name_vector))
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)

    for idx, (img, path) in enumerate(image_dataset):
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))

        print(f'inception features are : {idx}')
        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())


def augmented_features(file_pth, BATCH_SIZE=128):
    input_shape = (299, 299, 3)
    all_img_name_vector = dir_files(file_pth)
    inputs = keras.Input(shape=input_shape)
    flip_horizontal = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")(inputs)
    rnadom_contrast = tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.5)(inputs)

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
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)

    print('hello')
    for idx, (img, path, gray) in enumerate(image_dataset):

        bf_flip_inception, bf_contrast_inception = model(img)

        img_inception = tf.keras.applications.inception_v3.preprocess_input(img)
        gray_inception = tf.keras.applications.inception_v3.preprocess_input(gray)
        bf_flip_inception = tf.keras.applications.inception_v3.preprocess_input(bf_flip_inception)
        bf_contrast_inception = tf.keras.applications.inception_v3.preprocess_input(bf_contrast_inception)

        batch_features_inception = image_features_extract_model(img_inception)
        batch_features_inception = tf.reshape(batch_features_inception,
                                              (
                                                  batch_features_inception.shape[0], -1,
                                                  batch_features_inception.shape[3]))

        batch_features_FH = image_features_extract_model(bf_flip_inception)
        batch_features_FH = tf.reshape(batch_features_FH,
                                       (batch_features_FH.shape[0], -1, batch_features_FH.shape[3]))

        batch_features_CN = image_features_extract_model(bf_contrast_inception)
        batch_features_CN = tf.reshape(batch_features_CN,
                                       (batch_features_CN.shape[0], -1, batch_features_CN.shape[3]))

        batch_features_gray = image_features_extract_model(gray_inception)
        batch_features_gray = tf.reshape(batch_features_gray,
                                         (batch_features_gray.shape[0], -1, batch_features_gray.shape[3]))
        print(f'batch: {idx}')
        for FH, CN, p, inception, gray_features in zip(batch_features_FH, batch_features_CN, path,
                                                       batch_features_inception,
                                                       batch_features_gray):
            a = tf.stack([inception, FH, gray_features, CN])
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, a.numpy())

