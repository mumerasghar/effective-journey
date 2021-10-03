import tensorflow as tf
from tensorflow_addons.layers import SpectralNormalization
from .MultiHeadAttention import MultiHeadedAttention

# def critic_feed_forward(d_model, dff):
#     return tf.keras.Sequential(
#         [
#             SpectralNormalization(tf.keras.layers.Dense(dff)),
#             tf.keras.layers.LeakyReLU(alpha=0.1),
#             SpectralNormalization(tf.keras.layers.Dense(d_model, activation="sigmoid")),
#         ]
#     )
#
#
# class Critic(tf.keras.Model):
#     def __init__(self, d_output=1, d_input=512, rate=0.1):
#         super(Critic, self).__init__()
#
#         self.mha1 = MultiHeadedAttention(512, 8)
#         self.mha2 = MultiHeadedAttention(512, 2)
#
#         self.norm1 = SpectralNormalization(
#             tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         )
#         self.norm2 = SpectralNormalization(
#             tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         )
#         self.norm3 = SpectralNormalization(
#             tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         )
#
#         self.dropout1 = tf.keras.layers.Dropout(rate)
#         self.dropout2 = tf.keras.layers.Dropout(rate)
#         self.dropout3 = tf.keras.layers.Dropout(rate)
#
#         self.ffn = critic_feed_forward(d_output, d_input)
#
#     def call(self, x, training):
#         att1, _ = self.mha1(x, x, x)
#         out1 = self.dropout1(att1, training=training)
#         # out1 = self.norm1(att1 + x)
#
#         att2, _ = self.mha2(out1, out1, x)
#         out2 = self.dropout2(att2, training=training)
#         # out2 = self.norm2(att2 + x)
#
#         ffn_output = self.ffn(out2)
#         return ffn_output

xavier = tf.keras.initializers.GlorotNormal()
kaiming = tf.keras.initializers.HeNormal()


def critic_feed_forward(d_model, dff):
    return tf.keras.Sequential(
        [
            SpectralNormalization(tf.keras.layers.Dense(dff, kernel_initializer=kaiming)),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            SpectralNormalization(tf.keras.layers.Dense(d_model, activation="sigmoid", kernel_initializer=xavier)),
        ]
    )


class Critic(tf.keras.Model):
    def __init__(self, d_model, d_output=1, rate=0.1):
        super(Critic, self).__init__()

        self.mha1 = MultiHeadedAttention(d_model, 8)
        self.mha2 = MultiHeadedAttention(d_model, 2)

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

        self.ffn = critic_feed_forward(d_output, d_model)

    def call(self, x, training):
        att1, _ = self.mha1(x, x, x)
        out1 = self.dropout1(att1, training=training)
        # out1 = self.norm1(att1 + x)

        att2, _ = self.mha2(out1, out1, x)
        out2 = self.dropout2(att2, training=training)
        # out2 = self.norm2(att2 + x)

        ffn_output = self.ffn(out2)
        return ffn_output
