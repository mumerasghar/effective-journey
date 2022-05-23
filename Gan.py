from inference import evaluate
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras.layers import Conv2D
import tensorflow as tf

import os
import string
import time
import warnings

import numpy as np
import pandas as pd

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

warnings.filterwarnings("ignore")

image_path = "./Dataset/Flicker/Flicker8k_Dataset/"
dir_Flickr_text = "./Dataset/Flicker/Flickr8k.token.txt"

jpgs = os.listdir(image_path)
print(f"Total image in dataset is {len(jpgs)}")

file = open(dir_Flickr_text, "r")
text = file.read()
file.close()

datatxt = []
for line in text.split("\n"):
    col = line.split("\t")
    if len(col) == 1:
        continue

    w = col[0].split("#")
    datatxt.append(w + [col[1].lower()])

data = pd.DataFrame(datatxt, columns=["filename", "index", "captions"])
data = data.reindex(columns=["index", "filename", "captions"])
data = data[data.filename != "2258277193_586949ec62.jpg.1"]
uni_filenames = np.unique(data.filename.values)
data.head()

npic = 5
npix = 224
target_size = (npix, npix, 3)
count = 1

vocabulary = []
for txt in data.captions.values:
    vocabulary.extend(txt.split())

print(f"Vocublary Size {len(set(vocabulary))}")


def remove_punctutation(text_original):
    text_no_punctuation = text_original.translate(
        str.maketrans("", "", string.punctuation)
    )
    return text_no_punctuation


def remove_single_character(text):
    text_len_more_than1 = ""
    for word in text.split():
        if len(word) > 1:
            text_len_more_than1 += " " + word

    return text_len_more_than1


def remove_numeric(text):
    text_no_numeric = ""

    for word in text.split():
        isalpha = word.isalpha()

        if isalpha:
            text_no_numeric += " " + word

    return text_no_numeric


def text_clean(text_original):
    text = remove_punctutation(text_original)
    text = remove_single_character(text)
    text = remove_numeric(text)

    return text


for i, caption in enumerate(data.captions.values):
    newcaption = text_clean(caption)
    data["captions"].iloc[1] = newcaption

clean_vocab = []

for txt in data.captions.values:
    clean_vocab.extend(txt.split())

print(len(set(clean_vocab)))

all_captions = []

for caption in data["captions"].astype(str):
    caption = "<start> " + caption + " <end>"
    all_captions.append(caption)

all_img_name_vector = []
for annot in data["filename"]:
    full_image_path = image_path + annot
    all_img_name_vector.append(full_image_path)


def data_limiter(num, total_captions, all_img_name_vector):
    train_captions, img_name_vector = shuffle(
        total_captions, all_img_name_vector, random_state=1
    )
    train_captions = train_captions[:num]
    img_name_vector = img_name_vector[:num]
    return train_captions, img_name_vector


print("data limiter")
train_captions, img_name_vector = data_limiter(40000, all_captions, all_img_name_vector)


def load_image(image_path):
    print(image_path)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


# image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
# new_input = image_model.input
# hidden_layer = image_model.layers[-1].output
# image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# encode_train = sorted(set(img_name_vector))
# image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
# image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(64)

# for idx, (img, path) in enumerate(image_dataset):
#     batch_features = image_features_extract_model(img)
#     batch_features = tf.reshape(batch_features,
#                                 (batch_features.shape[0], -1, batch_features.shape[3]))
#     print(f'inception features batch_: {idx}')
#     for bf, p in zip(batch_features, path):
#         path_of_feature = p.numpy().decode("utf-8")
#         np.save(path_of_feature, bf.numpy())

top_k = 5000

print("tokenizer")
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=top_k, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ '
)
tokenizer.fit_on_texts(train_captions)

train_seqs = tokenizer.texts_to_sequences(train_captions)
tokenizer.word_index["<pad>"] = 0
train_seqs = tokenizer.texts_to_sequences(train_captions)
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding="post")

img_name_train, img_name_val, cap_train, cap_val = train_test_split(
    img_name_vector, cap_vector, test_size=0.2, random_state=0
)

BATCH_SIZE = 32
BUFFER_SIZE = 256
num_steps = len(img_name_train)


def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode("utf-8") + ".npy")
    return img_tensor, cap


print("creating dataste")
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
dataset = dataset.map(
    lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int32]
    ),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)
dataset = dataset.shuffle(128).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def v_load_pre_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (64, 64))
    return img


def v_map_func(img_name, cap):
    img_tensor = np.load(img_name.decode("utf-8") + ".npy")
    image = v_load_pre_image(img_name.decode("utf-8"))
    return img_tensor, cap, img_name, image


i_data = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))
i_data = i_data.map(
    lambda item1, item2: tf.numpy_function(
        v_map_func, [item1, item2], [tf.float32, tf.int32, tf.string, tf.float32]
    ),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)
i_data = i_data.shuffle(BUFFER_SIZE).batch(64)
i_data = i_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding_1d(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def positional_encoding_2d(row, col, d_model):
    assert d_model % 2 == 0
    row_pos = np.repeat(np.arange(row), col)[:, np.newaxis]
    col_pos = np.repeat(np.expand_dims(np.arange(col), 0), row, axis=0).reshape(-1, 1)

    angle_rads_row = get_angles(
        row_pos, np.arange(d_model // 2)[np.newaxis, :], d_model // 2
    )
    angle_rads_col = get_angles(
        col_pos, np.arange(d_model // 2)[np.newaxis, :], d_model // 2
    )

    angle_rads_row[:, 0::2] = np.sin(angle_rads_row[:, 0::2])
    angle_rads_row[:, 1::2] = np.cos(angle_rads_row[:, 1::2])

    angle_rads_col[:, 0::2] = np.sin(angle_rads_col[:, 0::2])
    angle_rads_col[:, 1::2] = np.cos(angle_rads_col[:, 1::2])

    pos_encoding = np.concatenate([angle_rads_row, angle_rads_col], axis=1)[
        np.newaxis, ...
    ]
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += mask * -1e9

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)
        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [tf.keras.layers.Dense(dff, activation="relu"), tf.keras.layers.Dense(d_model)]
    )


# class EncoderLayer(tf.keras.layers.Layer):

#     def __init__(self, d_model, num_heads, dff, rate=0.1):
#         super().__init__()
#         self.mha = MultiHeadedAttention(d_model, num_heads)
#         self.ffn = point_wise_feed_forward_network(d_model, dff)

#         self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

#         self.dropout1 = tf.keras.layers.Dropout(rate)
#         self.dropout2 = tf.keras.layers.Dropout(rate)

#     def call(self, x, training, mask=None):
#         attn_output, _ = self.mha(x, x, x, mask)
#         attn_output = self.dropout1(attn_output, training=training)

#         out1 = self.layernorm1(x + attn_output)

#         ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output, training=training)
#         out2 = self.layernorm2(out1 + ffn_output)

#         return out2


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

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


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
    def __init__(self, s, rate=0.1):
        super().__init__()

        self.block1 = ConvBlock(s)
        self.conv1x1_1 = SpectralNormalization(
            tf.keras.layers.Conv2D(s // 2, 1, use_bias=False)
        )
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.block2 = ConvBlock(s/2)
        self.conv1x1_2 = SpectralNormalization(
            tf.keras.layers.Conv2D(s // 4, 1, use_bias=False)
        )
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x = tf.reshape(x, (-1, 8, 8, 2048))
        activation = tf.keras.activations.relu

        _x = self.block1(x)
        x = _x + x
        x = self.layernorm1(activation(self.dropout1(self.conv1x1_1(x))))

        _x = self.block2(x)
        x = _x + x
        x = self.layernorm2(activation(self.dropout2(self.conv1x1_2(x))))

        x = tf.reshape(x, shape=(-1, 64, 512))
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self, num_layers, d_model, num_heads, dff, row_size, col_size, rate=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # self.embedding = (tf.keras.layers.Dense(self.d_model, activation="relu"))
        self.embedding = PreAttention(2048)
        # self.pos_encoding = positional_encoding_2d(row_size, col_size, self.d_model)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    # def call(self, x, training, mask=None):
    #     seq_len = tf.shape(x)[1]
    #     x = self.embedding(x)
    #     # x += self.pos_encoding[:, :seq_len, :]
    #     x = self.dropout(x, training=training)

    #     for i in range(self.num_layers):
    #         x = self.enc_layers[i](x, training, mask)

    #     return x
    def call(self, inp, training, mask=None):
        x = inp[:, 0, :, :]
        # x = inp
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        # x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            q = self.embedding(inp[:, i, :, :])
            x = self.enc_layers[i](x, x, q, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        target_vocab_size,
        maximum_position_encoding,
        rate=0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_embedding = positional_encoding_1d(maximum_position_encoding, d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask
            )
            attention_weights[f"decoder_layer{i + 1}_block1"] = block1
            attention_weights[f"decoder_layer{i + 2}_block2"] = block2

        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        row_size,
        col_size,
        target_vocab_size,
        max_pos_encoding,
        rate=0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff, row_size, col_size, rate
        )
        self.decoder = Decoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            target_vocab_size,
            max_pos_encoding,
            rate,
        )
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(
        self,
        inp,
        tar,
        training,
        look_ahead_mask=None,
        dec_padding_mask=None,
        enc_padding_mask=None,
    ):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask
        )
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights


num_layer = 4
d_model = 512
dff = 2048
num_heads = 8
row_size = 8
col_size = 8
# target_vocab_size = top_k + 1
target_vocab_size = 5000 + 1
dropout_rate = 0.1


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)
loss_mse = tf.keras.losses.BinaryCrossentropy()
optimizer_c = tf.keras.optimizers.Adam(0.0004, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def critic_feed_forward(d_model, dff):
    return tf.keras.Sequential(
        [
            SpectralNormalization(tf.keras.layers.Dense(dff)),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            SpectralNormalization(tf.keras.layers.Dense(d_model, activation="sigmoid")),
        ]
    )


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
        # ffn_output = self.dropout3(training=training)


train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name="train_accuracy")

transformer = Transformer(
    num_layer,
    d_model,
    num_heads,
    dff,
    row_size,
    col_size,
    target_vocab_size,
    max_pos_encoding=target_vocab_size,
    rate=dropout_rate,
)

critic = Critic()


def create_masks_decoder(tar):
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combined_mask


def dis_loss(f_cap, r_cap):
    b_shape = f_cap.shape[0]
    f_label = tf.zeros([b_shape, 1, 1])
    r_label = tf.ones([b_shape, 1, 1])

    r_output = critic(r_cap, True)
    # r_output = tf.reshape(r_output, shape=(b_shape))
    r_d_loss = loss_mse(r_label, r_output)
    r_d_loss = tf.reduce_sum(r_d_loss)

    f_output = critic(f_cap, True)
    # f_output = tf.reshape(f_output, shape=(b_shape))
    f_d_loss = loss_mse(f_label, f_output)
    f_d_loss = tf.reduce_sum(f_d_loss)

    return r_d_loss + f_d_loss


def gen_loss(tar_real, predictions, f_cap, r_cap):
    loss = loss_function(tar_real, predictions)

    b_shape = f_cap.shape[0]
    f_label = tf.zeros([b_shape, 1, 1])
    r_label = tf.ones([b_shape, 1, 1])

    g_output = critic(r_cap, True)
    # g_output = tf.reshape(g_output, shape=(b_shape))
    g_loss = loss_mse(r_label, g_output)
    g_loss = tf.reduce_sum(g_loss)

    return loss + g_loss


# @tf.function
def train_step(img_tensor, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    dec_mask = create_masks_decoder(tar_inp)

    with tf.GradientTape() as tape, tf.GradientTape() as d_tape:
        predictions, _ = transformer(img_tensor, tar_inp, True, dec_mask)
        f_cap = tf.argmax(predictions, axis=-1)

        loss = gen_loss(tar_real, predictions, f_cap, tar_real)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        d_loss = dis_loss(f_cap, tar_real)
        d_gradients = d_tape.gradient(d_loss, critic.trainable_variables)
        optimizer_c.apply_gradients(zip(d_gradients, critic.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


def generate_caption():
    for img_tensor, cap, img_name, image in i_data.take(1):
        f_cap, r_cap, name = evaluate(
            img_tensor, img_name, cap, tokenizer, transformer, show=False
        )
        return name, f_cap, r_cap


print("Going for train")


def main(epochs, o_break=False):
    print("going for training")

    checkpoint_dir = "checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        opt_transformer=optimizer,
        opt_discriminator=optimizer_c,
        transformer=transformer,
        critic=critic,
    )

    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

    transformer.save_weights("latest_weights")
    # if ckpt_manager.latest_checkpoint:
    #     checkpoint.restore(ckpt_manager.latest_checkpoint)
    #     print("Latest checkpoint restored!!")

    for epoch in range(epochs):

        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (img_tensor, tar)) in enumerate(dataset):
            train_step(img_tensor, tar)

            if batch % 50 == 0:
                print(
                    f"Epoch {epoch + 1}, Batch {batch}, Loss {train_loss.result()}, Accuracy {train_accuracy.result():.4f}"
                )
            if o_break:
                break

        if o_break:
                break

        with open("result.txt", "a") as f:
            name, f_cap, r_cap = generate_caption()
            f.write(
                f"Epoch {epoch + 1}, Batch {batch}, Loss {train_loss.result()}, Accuracy {train_accuracy.result():.4f}\n"
            )
            f.write(f"Time taken for 1 epoch : {time.time() - start} secs\n\n\n")
            f.write(f"img_name:{name},\nr_cap: {r_cap}\nfake: {f_cap}\n\n")

        print(
            f"Epoch {epoch + 1}, Batch {batch}, Loss {train_loss.result()}, Accuracy {train_accuracy.result():.4f}"
        )
        print(f"Time taken for 1 epoch : {time.time() - start} secs\n\n\n")

        ckpt_save_path = ckpt_manager.save()
    transformer.save_weights(f"checkpoints/transformer_final_weights_{epoch}")


if __name__ == "__main__":
    # main(30)
    main(30, False)

else:

    main(1, True)
    print("loading transformer weights")
    transformer.load_weights("checkpoints/transformer_final_weights")
    # while True:
    #     generate_caption()
