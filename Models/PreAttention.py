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
    def __init__(self, s, o, rate=0.1):
        super().__init__()

        self.block1 = ConvBlock(s)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.gamma = tf.Variable(tf.zeros(1), trainable=True)

        self.conv1x1 = SpectralNormalization(tf.keras.layers.Conv2D(o, 1, use_bias=False))

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
    def __init__(self, s, d_model, convert_dim=False):
        super().__init__()

        self.d_model = d_model
        if convert_dim:
            self.dim = 2048
            self.model = tf.keras.Sequential([
                PreAttention(s, 1024),
                PreAttention(1024, self.d_model),
            ])
        else:
            self.dim = d_model
            self.model = tf.keras.Sequential([
                PreAttention(self.d_model, self.d_model),
                PreAttention(self.d_model, self.d_model)
            ])

    def call(self, x):
        s = int(x.shape[1] ** (1 / 2))
        x = tf.reshape(x, (-1, s, s, self.dim))
        x = self.model(x)
        x = tf.reshape(x, shape=(-1, s * s, self.d_model))
        return x
