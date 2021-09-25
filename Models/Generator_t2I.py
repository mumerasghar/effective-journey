import pickle
from random import randint

import matplotlib.pyplot as pyplot
import numpy as np
import tensorflow as tf
from keras.preprocessing.image class PreAttention(tf.keras.layers.Layer):
    def __init__(self, s, keep_dims=False, rate=0.1):
        super().__init__()

        self.block1 = ConvBlock(s)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.gamma = tf.Variable(tf.zeros(1), trainable=True)

        if not keep_dims:
            self.conv1x1 = SpectralNormalization(tf.keras.layers.Conv2D(s // 2, 1, use_bias=False))
        else:
            self.conv1x1 = SpectralNormalization(tf.keras.layers.Conv2D(s, 1, use_bias=False))

    def call(self, x, training=True):
        attn_out = self.block1(x)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.layernorm1(self.gamma * attn_out + x)

        out1 = self.conv1x1(out1)
        out1 = self.dropout2(out1, training=training)
        out1 = tf.keras.activations.relu(out1)
        out1 = self.layernorm2(out1)

        return out1


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, s, rate=0.1, training=True):
        super().__init__()

        self.p_attention1 = PreAttention(s)
        self.p_attention2 = PreAttention(s // 2)
        self.p_attention3 = PreAttention(s // 4, keep_dims=True)
        self.p_attention4 = PreAttention(s // 4, keep_dims=True)

    def call(self, x):
        x = tf.reshape(x, (-1, 8, 8, 2048))

        x = self.p_attention1(x)
        x = self.p_attention2(x)
        x = self.p_attention3(x)
        x = self.p_attention4(x)

        x = tf.reshape(x, shape=(-1, 64, 512))
        return ximport array_to_img
from numpy import asarray
from numpy import expand_dims
from numpy.random import randint
from tensorflow.keras import Model
from tensorflow.keras import layers


# model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',
#                                                         binary=True)


def random_flip(image):
    image = tf.image.flip_left_right(image)
    return image.numpy()


def random_jitter(image):
    # add additional dimension necessary for zooming
    image = expand_dims(image, 0)
    image = image_augmentation_generator.flow(image, batch_size=1)
    # remove additional dimension (1, 64, 64, 3) to (64, 64, 3)
    result = image[0].reshape(image[0].shape[1:])
    return result


image_augmentation_generator = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=[0.8,
                                                                                           1.0])  # random zoom proves to be helpful in capturing more details https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/


def get_random_word_vectors_from_dataset(n_samples, captions):
    ix = np.random.randint(0, len(captions), n_samples)
    return np.asarray(captions)[ix]


def generate_random_vectors(n_samples):
    vectorized_random_captions = []
    for n in range(n_samples):
        vectorized_random_captions.append(
            tf.random.uniform([300]))
    return vectorized_random_captions


# Discriminator model
def define_discriminator():
    word_vector_dim = 300
    dropout_prob = 0.4

    in_label = layers.Input(shape=(300,))

    n_nodes = 3 * 64 * 64
    li = layers.Dense(n_nodes)(in_label)
    li = layers.Reshape((64, 64, 3))(li)

    dis_input = layers.Input(shape=(64, 64, 3))

    merge = layers.Concatenate()([dis_input, li])

    discriminator = layers.Conv2D(
        filters=64, kernel_size=(3, 3), padding="same")(merge)
    discriminator = layers.LeakyReLU(0.2)(discriminator)
    discriminator = layers.GaussianNoise(0.2)(discriminator)

    discriminator = layers.Conv2D(filters=64, kernel_size=(
        3, 3), strides=(2, 2), padding="same")(discriminator)
    discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
    discriminator = layers.LeakyReLU()(discriminator)

    discriminator = layers.Conv2D(filters=128, kernel_size=(
        3, 3), padding="same")(discriminator)
    discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
    discriminator = layers.LeakyReLU(0.2)(discriminator)

    discriminator = layers.Conv2D(filters=128, kernel_size=(
        3, 3), strides=(2, 2), padding="same")(discriminator)
    discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
    discriminator = layers.LeakyReLU(0.2)(discriminator)

    discriminator = layers.Conv2D(filters=256, kernel_size=(
        3, 3), padding="same")(discriminator)
    discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
    discriminator = layers.LeakyReLU(0.2)(discriminator)

    discriminator = layers.Conv2D(filters=256, kernel_size=(
        3, 3), strides=(2, 2), padding="same")(discriminator)
    discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
    discriminator = layers.LeakyReLU(0.2)(discriminator)

    discriminator = layers.Conv2D(filters=512, kernel_size=(
        3, 3), padding="same")(discriminator)
    discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
    discriminator = layers.LeakyReLU(0.2)(discriminator)

    discriminator = layers.Flatten()(discriminator)

    discriminator = layers.Dense(1024)(discriminator)

    discriminator = layers.LeakyReLU(0.2)(discriminator)

    discriminator = layers.Dense(1)(discriminator)

    discriminator_model = Model(
        inputs=[dis_input, in_label], outputs=discriminator)

    # discriminator_model.summary()

    return discriminator_model


def resnet_block(model, kernel_size, filters, strides):
    gen = model
    model = layers.Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding="same")(model)
    model = layers.BatchNormalization(momentum=0.5)(model)
    model = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                                  shared_axes=[1, 2])(model)
    model = layers.Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding="same")(model)
    model = layers.BatchNormalization(momentum=0.5)(model)
    model = layers.Add()([gen, model])
    return model


# Generator model
def define_generator():
    kernel_init = tf.random_normal_initializer(stddev=0.02)
    batch_init = tf.random_normal_initializer(1., 0.02)

    random_input = layers.Input(shape=(100,))
    text_input1 = layers.Input(shape=(300,))
    text_layer1 = layers.Dense(8192)(text_input1)
    text_layer1 = layers.Reshape((8, 8, 128))(text_layer1)

    n_nodes = 128 * 8 * 8
    gen_input_dense = layers.Dense(n_nodes)(random_input)
    generator = layers.Reshape((8, 8, 128))(gen_input_dense)

    merge = layers.Concatenate()([generator, text_layer1])

    model = layers.Conv2D(filters=64, kernel_size=9,
                          strides=1, padding="same")(merge)
    model = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                                  shared_axes=[1, 2])(model)

    gen_model = model

    for _ in range(4):
        model = resnet_block(model, 3, 64, 1)

    model = layers.Conv2D(filters=64, kernel_size=3,
                          strides=1, padding="same")(model)
    model = layers.BatchNormalization(momentum=0.5)(model)
    model = layers.Add()([gen_model, model])

    model = layers.Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same",
                                   kernel_initializer=kernel_init)(model)
    model = layers.LeakyReLU(0.2)(model)

    model = layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same",
                                   kernel_initializer=kernel_init)(model)
    model = layers.LeakyReLU(0.2)(model)

    model = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same",
                                   kernel_initializer=kernel_init)(model)
    model = layers.LeakyReLU(0.2)(model)

    model = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                   kernel_initializer=kernel_init)(model)
    model = layers.LeakyReLU(0.2)(model)

    model = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(model)

    generator_model = Model(inputs=[random_input, text_input1], outputs=model)

    # generator_model.summary()
    tf.keras.utils.plot_model(generator_model, to_file='model.png', show_shapes=True)
    return generator_model


def generate_latent_points(latent_dim, n_samples, captions):
    x_input = tf.random.normal([n_samples, latent_dim])
    text_captions = get_random_word_vectors_from_dataset(n_samples, captions)
    return [x_input, text_captions]


# Randomly flip some labels. Credits to https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/
def noisy_labels(y, p_flip):
    n_select = int(p_flip * int(y.shape[0]))
    flip_ix = np.random.choice(
        [i for i in range(int(y.shape[0]))], size=n_select)

    op_list = []
    for i in range(int(y.shape[0])):
        if i in flip_ix:
            op_list.append(tf.subtract(1.0, y[i]))
        else:
            op_list.append(y[i])

    outputs = tf.stack(op_list)
    return outputs


def smooth_positive_labels(y):
    return y - 0.3 + (np.random.random(y.shape) * 0.5)


def smooth_negative_labels(y):
    return y + np.random.random(y.shape) * 0.3


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    print(predictions.shape)
    pyplot.figure(figsize=[7, 7])

    for i in range(predictions.shape[0]):
        pyplot.subplot(5, 5, i + 1)
        pyplot.imshow(array_to_img(predictions.numpy()[i]))
        pyplot.axis('off')

    pyplot.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # pyplot.show()


def t2I_discriminator_loss(r_real_output_real_text, f_fake_output_real_text_1, f_real_output_fake_text):
    alpha = 0.5
    real_output_noise = smooth_positive_labels(
        noisy_labels(tf.ones_like(r_real_output_real_text), 0.10))
    fake_output_real_text_noise_1 = smooth_negative_labels(
        tf.zeros_like(f_fake_output_real_text_1))
    real_output_fake_text_noise = smooth_negative_labels(
        tf.zeros_like(f_real_output_fake_text))

    real_loss = tf.reduce_mean(loss_mse(
        real_output_noise, r_real_output_real_text))
    fake_loss_ms_1 = tf.reduce_mean(loss_mse(
        fake_output_real_text_noise_1, f_fake_output_real_text_1))
    fake_loss_2 = tf.reduce_mean(loss_mse(
        real_output_fake_text_noise, f_real_output_fake_text))

    total_loss = real_loss + alpha * fake_loss_2 + (1 - alpha) * fake_loss_ms_1
    return total_loss


def t2I_generator_loss(f_fake_output_real_text):
    return tf.reduce_mean(loss_mse(tf.ones_like(f_fake_output_real_text), f_fake_output_real_text))


class TextEncode(tf.keras.Model):
    def __init__(self, vocab_size, out_dim=300):
        super().__init__()
        self.emb = layers.Embedding(input_dim=vocab_size, output_dim=out_dim)
        self.rnn = layers.LSTM(300)

    def call(self, x):
        x = self.emb(x)
        return self.rnn(x)


loss_mse = tf.keras.losses.BinaryCrossentropy(from_logits=True)
