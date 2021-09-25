import tensorflow as tf
from .Generator_i2T import Encoder, MultiHeadedAttention


def critic_feed_forward(d_model, output):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(d_model, kernel_initializer='glorot_uniform'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dense(output, kernel_initializer='glorot_uniform')
    ])


class Critic(tf.keras.Model):

    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Critic, self).__init__()
        self.enc = Encoder(num_layers, d_model, num_heads, dff, rate)
        self.ffn = critic_feed_forward(512, 1)

    def call(self, x, training, mask=None):
        x = self.enc(x, training)
        return self.ffn(x)
