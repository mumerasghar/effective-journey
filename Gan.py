import os
import time

import yaml
from tensorflow_addons.layers import SpectralNormalization

from data import create_dataset
from inference import evaluate
from inference import karpathy_inference

from utils import *

DATASET = 'COCO_RCNN'
with open('./cfg/cfg.yaml', 'r') as f:
    cfg = yaml.load(f)
    cfg = cfg[DATASET]

paths = {
    "image_path": cfg['IMG_PATH'],
    "text_path": cfg["TXT_PATH"],
    "cap_file": cfg["CAP_FILE"],
    "img_name": cfg["IMG_NAME"],
    "dataset": cfg["DATASET_NAME"]
}

dataset, i_data, tokenizer = create_dataset(cfg)

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
        [
            tf.keras.layers.Dense(dff, activation="relu"),
            tf.keras.layers.Dense(d_model),
        ]
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
        # self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.layernorm1(x + attn_output)

        # ffn_output = self.ffn(out1)
        # ffn_output = self.dropout2(ffn_output, training=training)
        # out2 = self.layernorm2(out1 + ffn_output)

        return out1


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
    def __init__(self, s, convert_dim=False):
        super().__init__()

        if convert_dim:
            self.dim = 2048
            self.model = tf.keras.Sequential([
                PreAttention(s),
                PreAttention(s // 2),
            ])
        else:
            self.dim = 512
            self.model = tf.keras.Sequential([
                PreAttention(s // 4, keep_dims=True),
                PreAttention(s // 4, keep_dims=True)
            ])

    def call(self, x):
        x = tf.reshape(x, (-1, 8, 8, self.dim))
        x = self.model(x)
        x = tf.reshape(x, shape=(-1, 64, 512))
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = (tf.keras.layers.Dense(self.d_model, activation="relu"))
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

        # self.s_attention = []
        # for _ in range(num_layers):
        #     if _ == 0:
        #         self.s_attention.append(SelfAttention(dff, convert_dim=True))
        #     else:
        #         self.s_attention.append(SelfAttention(dff))
        #
        self.dropout = tf.keras.layers.Dropout(rate)
        # # skeptical of being using relu here.
        # self.conv_net = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(d_model, (3, 3), padding='same', activation='relu')
        # ])

    def call(self, x, training, mask=None):
        inp = x
        x = self.embedding(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            # if i == 0:
            #     _atn_module = self.s_attention[i](inp)
            # else:
            #     _atn_module = self.s_attention[i](x)
            # _enc_out = self.enc_layers[i](x, training, mask)
            # _x = tf.concat([_atn_module, _enc_out], axis=-1)
            # _x = tf.reshape(_x, (-1, 8, 8, 1024))
            # _x = self.conv_net(_x)
            # x = tf.reshape(_x, (-1, 64, 512))
            x = self.enc_layers[i](x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1, ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_embedding = positional_encoding(maximum_position_encoding, d_model)

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
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, max_pos_encoding,
                 rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)
        # self.encoder = SelfAttention(2048)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, max_pos_encoding, rate, )
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, look_ahead_mask=None, dec_padding_mask=None, enc_padding_mask=None, ):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        # enc_output = self.encoder(inp)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights


num_layer = 4
d_model = 512
dff = 2048
num_heads = 8
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

transformer = Transformer(num_layer, d_model, num_heads, dff, cfg['VOCAB_SIZE'],
                          max_pos_encoding=cfg['VOCAB_SIZE'], rate=dropout_rate)

critic = Critic()


def create_masks_decoder(tar):
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combined_mask


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def dis_loss(f_cap, r_cap):
    b_shape = f_cap.shape[0]
    # f_label = tf.zeros([b_shape, 1, 1])
    # r_label = tf.ones([b_shape, 1, 1])

    r_output = critic(r_cap, True)
    # r_output = tf.reshape(r_output, shape=(b_shape))
    r_d_loss = loss_mse(tf.ones_like(r_output), r_output)
    r_d_loss = tf.reduce_sum(r_d_loss)

    f_output = critic(f_cap, True)
    # f_output = tf.reshape(f_output, shape=(b_shape))
    f_d_loss = loss_mse(tf.zeros_like(f_output), f_output)
    f_d_loss = tf.reduce_sum(f_d_loss)

    return r_d_loss + f_d_loss


def gen_loss(tar_real, predictions, f_cap, r_cap):
    loss = loss_function(tar_real, predictions)

    b_shape = f_cap.shape[0]
    # f_label = tf.zeros([b_shape, 1, 1])
    # r_label = tf.ones([b_shape, 1, 1])

    g_output = critic(r_cap, True)
    # g_output = tf.reshape(g_output, shape=(b_shape))
    # g_loss = loss_mse(r_label, g_output)
    g_loss = loss_mse(tf.ones_like(g_output), g_output)
    g_loss = tf.reduce_sum(g_loss)

    return loss + g_loss


@tf.function
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
    for img_tensor, cap, img_name in i_data.take(1):
        f_cap, r_cap, names = evaluate(img_tensor, img_name, cap, tokenizer, transformer, show=False)
        return names, f_cap, r_cap


def checkpoint_manager():
    checkpoint_dir = "checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        opt_transformer=optimizer,
        opt_discriminator=optimizer_c,
        transformer=transformer,
        critic=critic,
    )

    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")

    return ckpt_manager


def main(epochs, o_break=False):
    print("going for training")
    ckpt_manager = checkpoint_manager()

    for epoch in range(epochs):

        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (img_tensor, tar,_)) in enumerate(dataset):
            train_step(img_tensor, tar)

            if batch % 100 == 0:
                print(
                    f"Epoch {epoch + 1}, Batch {batch}, Loss {train_loss.result()}, Accuracy {train_accuracy.result():.4f}"
                )

            if o_break:
                return

        if o_break:
            return

        log_time = f"Time taken for 1 epoch : {time.time() - start} secs"
        log_accuracy = f"Epoch {epoch + 1}, Batch {batch}, Loss {train_loss.result()}, Accuracy {train_accuracy.result():.4f}"

        with open("result.txt", "a") as f:
            name, f_cap, r_cap = generate_caption()
            f.write(f"{log_accuracy}\n")
            f.write(f"{log_time}\n\n")
            f.write(f"img_name:{name},\nr_cap: {r_cap}\nfake: {f_cap}\n")
            f.write("-" * 100 + "\n")

        ckpt_save_path = ckpt_manager.save()

        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
        print(f'{log_accuracy}')
        print(f'{log_time}\n')


if __name__ == "__main__":
    with open('result.txt', 'w') as file:
        file.write('-'*45+" GANS "+"-"*45+'\n')
        file.write('-'*35+f" DATASET_SIZE: [{cfg['DATASET_SIZE']}]"+"-"*35+'\n')
        file.write('-'*35+f" VOCAB_SIZE: [{cfg['VOCAB_SIZE']}]"+"-"*35+'\n')
        file.write('-'*35+f" BATCH_SIZE: [{cfg['BATCH_SIZE']}]"+"-"*35+'\n')
        file.write('-'*35+f" EPOCHS: [{cfg['EPOCHS']}]"+"-"*35+'\n')
        file.write('-'*35+f" DATASET_NAME: [{cfg['DATASET_NAME']}]"+"-"*35+'\n')
        file.write('-' * 100 + '\n')
    main(cfg['EPOCHS'], False)
    karpathy_inference(tokenizer, transformer,cfg)

else:
    from inference import karpathy_inference

    checkpoint_manager()
    karpathy_inference(tokenizer, transformer,cfg)
