from Models import *
from data.create_dataset import create_dataset, tokenizer
from inference import evaluate

import os
import time
import warnings

warnings.filterwarnings("ignore")

NUM_LAYERS = 4
D_MODEL = 512
DFF = 2048
NUM_HEADS = 8
BATCH_SIZE = 64
CRITIC_ITERATIONS = 2
LAMBDA = 10
TARGET_VOCAB_SIZE = 5000 + 1
DROPOUT_RATE = 0.1
ROW_SIZE = 8
COL_SIZE = 8
LATENT_DIMENSION = 100


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


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


# ###################################### TRAINING FUNCTIONS #########################################

# @tf.function
def train_step(img_tensor, tar, img_name, img):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    dec_mask = create_masks_decoder(tar_inp)

    with tf.GradientTape() as tape, tf.GradientTape() as d_tape:
        predictions, _ = transformer(img_tensor, tar_inp, True, dec_mask)
        f_cap = tf.argmax(predictions, axis=-1)

        loss = gen_loss(tar_real, predictions, f_cap, tar_real)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer_g.apply_gradients(zip(gradients, transformer.trainable_variables))

        d_loss = dis_loss(f_cap, tar_real)
        d_gradients = d_tape.gradient(d_loss, critic.trainable_variables)
        optimizer_c.apply_gradients(zip(d_gradients, critic.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


def generate_caption():
    for img_tensor, cap, img_name, image in i_dataset.take(1):
        f_cap, r_cap, names = evaluate(img_tensor, img_name, cap, tokenizer, transformer, show=False)
        return names, f_cap, r_cap


def checkpoint():
    checkpoint_dir = 'checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        i2T_generator_optimizer=optimizer_g,
        i2T_discriminator_optimizer=optimizer_c,
        i2T_generator=transformer,
        i2T_discriminator=critic)

    ckpt_manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=1)

    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    return ckpt_manager


def train(dataset, epochs, t_break=False):
    for epoch in range(epochs):

        ckpt_manager = checkpoint()

        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (img_tensor, tar, img_name, img)) in enumerate(dataset):
            train_step(img_tensor, tar, img_name, img)

            if batch % 50 == 0:
                print(
                    f'Epoch {epoch + 1}, Batch {batch}, Loss {train_loss.result()}, Accuracy {train_accuracy.result():.4f}')
            if t_break:
                return

        log_time = f"Time taken for 1 epoch : {time.time() - start} secs"
        log_accuracy = f"Epoch {epoch + 1}, Batch {batch}, Loss {train_loss.result()}, Accuracy {train_accuracy.result():.4f}"

        with open("result.txt", "a") as f:
            name, f_cap, r_cap = generate_caption()
            f.write(f"\n{log_time}\n")
            f.write(f"{log_accuracy}\n")
            f.write(f"img_name:{name},\nr_cap: {r_cap}\nfake: {f_cap}\n\n")
            f.write("-- " * 100 + "\n")

        print(f'{log_accuracy}')
        print(f'{log_time}\n')

        # ### saving checkpoint
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))


# ################################  IMAGE2TEXT NETWORK AND OPTIMIZER ################################

learning_rate = CustomSchedule(D_MODEL)
transformer = Transformer(NUM_LAYERS, D_MODEL, NUM_HEADS, DFF, TARGET_VOCAB_SIZE,
                          max_pos_encoding=TARGET_VOCAB_SIZE, rate=DROPOUT_RATE)
critic = Critic(NUM_LAYERS, D_MODEL, NUM_HEADS, DFF)

optimizer_g = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
loss_mse = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer_c = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name='train_accuracy')

dataset, i_dataset = create_dataset()

if __name__ == '__main__':
    train(dataset, 30)
else:
    checkpoint()
