import os
import numpy as np
import tensorflow as tf

# from clean_data import all_img_name_vector


FILE_PATH = "./Dataset/COCO/extracted/train2014/"
BATCH_SIZE = 128

_files = os.listdir(FILE_PATH)
all_img_name_vector = [i for i in _files if not i.endswith('.npy')]
all_img_name_vector = [FILE_PATH + i for i in all_img_name_vector]


def load_image(image_path):
    print(image_path)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


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
