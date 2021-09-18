from Models import *
from data.create_dataset import create_dataset, tokenizer
from inference import evaluate

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


def i2T_dis_loss(f_cap, r_cap):
    b_shape = f_cap.shape[0]

    r_cap = tf.reshape(r_cap, shape=(b_shape, 1, -1))
    r_output = i2T_critic(r_cap, True)

    r_d_loss = loss_mse(tf.ones_like(r_output), r_output)
    r_d_loss = tf.reduce_sum(r_d_loss)

    f_cap = tf.reshape(f_cap, shape=(b_shape, 1, -1))
    f_output = i2T_critic(f_cap, True)

    f_d_loss = loss_mse(tf.zeros_like(f_output), f_output)
    f_d_loss = tf.reduce_sum(f_d_loss)

    return r_d_loss + f_d_loss


def i2T_gen_loss(tar_real, predictions, f_cap, r_cap):
    loss = i2T_loss_function(tar_real, predictions)
    g_loss = 0
    # b_shape = f_cap.shape[0]
    # r_cap = tf.reshape(r_cap, shape=(b_shape, 1, -1))
    # g_output = i2T_critic(r_cap, True)
    #
    # g_loss = loss_mse(tf.ones_like(g_output), g_output)
    # g_loss = tf.reduce_sum(g_loss)

    return loss + g_loss


def i2T_loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


# ###################################### TRAINING FUNCTIONS #########################################

@tf.function
def train_step(img_tensor, tar, img_name, img):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    dec_mask = create_masks_decoder(tar_inp)
    # tf.GradientTape() as d_tape,
    # tf.GradientTape() as gen_tape,
    # tf.GradientTape() as disc_tape,
    # tf.GradientTape() as rnn,
    # tf.GradientTape() as d_tape,
    # tf.GradientTape() as gen_tape,
    # tf.GradientTape() as disc_tape,
    # tf.GradientTape() as rnn

    with tf.GradientTape() as tape:
        # predictions, _ = transformer(img_tensor, tar_inp, True, dec_mask)
        predictions, _ = i2T_generator(img_tensor, tar_inp, True, dec_mask)
        f_cap = tf.argmax(predictions, axis=-1)

        loss = i2T_gen_loss(tar_real, predictions, f_cap, tar_real)
        # d_loss = i2T_dis_loss(f_cap, tar_real)

    # d_gradients = d_tape.gradient(d_loss, i2T_critic.trainable_variables)
    gradients = tape.gradient(loss, i2T_generator.trainable_variables)

    # i2T_c_optimizer.apply_gradients(zip(d_gradients, i2T_critic.trainable_variables))
    i2T_g_optimizer.apply_gradients(zip(gradients, i2T_generator.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


def generate_caption():
    for img_tensor, cap, img_name, image in i_data.take(1):
        evaluate(img_tensor, img_name, cap, tokenizer, i2T_generator, show=False)
        break


def train(dataset, epochs, t_break=False):
    checkpoint_dir = 'checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        i2T_generator_optimizer=i2T_g_optimizer,
        i2T_discriminator_optimizer=i2T_c_optimizer,
        i2T_generator=i2T_generator,
        i2T_discriminator=i2T_critic)

    ckpt_manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=1)

    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    for epoch in range(epochs):

        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (img_tensor, tar, img_name, img)) in enumerate(dataset):
            train_step(img_tensor, tar, img_name, img)

            if batch % 50 == 0:
                print(
                    f'Epoch {epoch + 1}, Batch {batch}, Loss {train_loss.result()}, Accuracy {train_accuracy.result():.4f}')

            if t_break:
                break

        if t_break:
            break
        generate_caption()
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

        print(f'Epoch {epoch + 1}, Batch {batch}, Loss {train_loss.result()}, Accuracy {train_accuracy.result():.4f}')
        print(f'Time taken for 1 epoch : {time.time() - start} secs\n')


# ################################  IMAGE2TEXT NETWORK AND OPTIMIZER ################################

learning_rate = CustomSchedule(D_MODEL)
i2T_generator = Transformer(NUM_LAYERS, D_MODEL, NUM_HEADS, DFF, TARGET_VOCAB_SIZE,
                            max_pos_encoding=TARGET_VOCAB_SIZE, rate=DROPOUT_RATE)
i2T_critic = Critic(NUM_LAYERS, D_MODEL, NUM_HEADS, DFF)

i2T_g_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
loss_mse = tf.keras.losses.BinaryCrossentropy(from_logits=True)
i2T_c_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name='train_accuracy')
dataset, i_dataset = create_dataset()

if __name__ == '__main__':
    train(dataset, 30)
else:
    train(i_dataset, 1, True)
