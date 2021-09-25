import tensorflow as tf
from tensorflow_addons.layers import SpectralNormalization


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, s):
        super().__init__()

        self.f_x = tf.keras.layers.Conv2D(s // 8, 1)
        self.g_x = tf.keras.layers.Conv2D(s // 8, 1)
        self.h_x = tf.keras.layers.Conv2D(s, 1)

    def call(self, x):
        f_x = self.f_x(x)
        g_x = self.g_x(x)
        h_x = self.h_x(x)

        fg_x = tf.linalg.matmul(f_x, g_x, transpose_b=True)
        fg_x = tf.keras.activations.sigmoid(fg_x)

        out = tf.linalg.matmul(fg_x, h_x)
        return out


class PreAttention(tf.keras.layers.Layer):
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
        return x
