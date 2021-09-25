import tensorflow as tf
from tensorflow_addons.layers import SpectralNormalization
from .Generator import Encoder, MultiHeadedAttention


def critic_feed_forward(d_model, output):
    return tf.keras.Sequential([
        SpectralNormalization(tf.keras.layers.Dense(d_model, kernel_initializer='glorot_uniform')),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dense(output, kernel_initializer='glorot_uniform')
    ])


# class CriticEnc()

class Critic(tf.keras.Model):

    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Critic, self).__init__()
        self.enc = Encoder(num_layers, d_model, num_heads, dff, rate)
        self.ffn = critic_feed_forward(512, 1)

    def call(self, x, training, mask=None):
        x = tf.reshape(x, shape=(x.shape[0], 1, x.shape[1]))
        x = self.enc(x, training)
        return self.ffn(x)
