from utils import *
from .PreAttention import *
from .MultiHeadAttention import MultiHeadedAttention
from tensorflow_addons.layers import SpectralNormalization

xavier = tf.keras.initializers.GlorotNormal()
kaiming = tf.keras.initializers.HeNormal()


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu', kernel_initializer=kaiming),
        tf.keras.layers.Dense(d_model, kernel_initializer=xavier)
    ])


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha = MultiHeadedAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, v, k, q, training, mask=None):
        attn_output, _ = self.mha(v, k, q, mask)
        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.layernorm1(v + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha1 = MultiHeadedAttention(d_model, num_heads)
        self.mha2 = MultiHeadedAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


# class Encoder(tf.keras.layers.Layer):
#
#     def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.num_layers = num_layers
#
#         self.embedding = SpectralNormalization(
#             tf.keras.layers.Dense(self.d_model,
#                                   activation='relu',
#                                   kernel_initializer=kaiming))
#
#         self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
#         self.dropout = tf.keras.layers.Dropout(rate)
#
#     def call(self, inp, training, mask=None):
#         x = inp
#         x = self.embedding(x)
#         x = self.dropout(x, training=training)
#         for i in range(self.num_layers):
#             x = self.enc_layers[i](x, x, x, training, mask)
#         return x

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = (tf.keras.layers.Dense(self.d_model, activation="relu"))
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

        self.s_attention = []
        for _ in range(num_layers):
            if _ == 0:
                self.s_attention.append(SelfAttention(dff, d_model, convert_dim=True))
            else:
                self.s_attention.append(SelfAttention(d_model, d_model))

        self.dropout = tf.keras.layers.Dropout(rate)
        # skeptical of being using relu here.
        self.conv_net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(d_model, (3, 3), padding='same', activation='relu')
        ])

    def call(self, x, training, mask=None):

        inp = x
        x = self.embedding(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            if i == 0:
                _atn_module = self.s_attention[i](inp)
            else:
                _atn_module = self.s_attention[i](x)
            _enc_out = self.enc_layers[i](x, training, mask)
            _x = tf.concat([_atn_module, _enc_out], axis=-1)
            _x = tf.reshape(_x, (-1, 8, 8, self.d_model * 2))
            _x = self.conv_net(_x)
            x = tf.reshape(_x, (-1, 64, self.d_model))

        return x


class Decoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_embedding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 2}_block2'] = block2

        return x, attention_weights


class Transformer(tf.keras.Model):

    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, max_pos_encoding,
                 rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, max_pos_encoding, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size, kernel_initializer='glorot_uniform')

    def call(self, inp, tar, training, look_ahead_mask=None, dec_padding_mask=None, enc_padding_mask=None):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights
