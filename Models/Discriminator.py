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

# class Critic(tf.keras.Model):
#
#     def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
#         super(Critic, self).__init__()
#         self.enc = Encoder(num_layers, d_model, num_heads, dff, rate)
#         self.ffn = critic_feed_forward(512, 1)
#
#     def call(self, x, training, mask=None):
#         x = tf.reshape(x, shape=(x.shape[0], 1, x.shape[1]))
#         x = self.enc(x, training)
#         return self.ffn(x)


class Critic(tf.keras.Model):
    def __init__(self, d_output=1, d_input=512, rate=0.1):
        super(Critic, self).__init__()

        self.mha1 = MultiHeadedAttention(512, 8)
        self.mha2 = MultiHeadedAttention(512, 2)

        self.norm1 = SpectralNormalization(
            tf.keras.layers.LayerNormalization(epsilon=1e-6)
        )
        self.norm2 = SpectralNormalization(
            tf.keras.layers.LayerNormalization(epsilon=1e-6)
        )
        self.norm3 = SpectralNormalization(
            tf.keras.layers.LayerNormalization(epsilon=1e-6)
        )

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

        self.ffn = critic_feed_forward(d_output, d_input)

    def call(self, x, training):
        att1, _ = self.mha1(x, x, x)
        out1 = self.dropout1(att1, training=training)
        # out1 = self.norm1(att1 + x)

        att2, _ = self.mha2(out1, out1, x)
        out2 = self.dropout2(att2, training=training)
        # out2 = self.norm2(att2 + x)

        ffn_output = self.ffn(out2)
        return ffn_output
