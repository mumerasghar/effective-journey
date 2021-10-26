import os
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

# matplotlib.use("TkAgg")
results = {}
tokenizer = None
# transformer = None
image_path = './Dataset/Flicker8k_Dataset/'
dir_Flickr_text = './Dataset/Flickr8k.token.txt'


def i_map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, img_name, cap


def remove_list_extension(i):
    ret_val, *_ = i.split('.')
    return ret_val


def append_to_list(id, name):
    name = name.decode('utf-8')
    word = tokenizer.index_word[int(id[0])]
    if name in results.keys():
        results[name].append(word)
    else:
        results[name] = [word]

    return results[name]


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_masks_decoder(tar):
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combined_mask


def evaluate(image, names, cap_real, tokenize, transformer,img_rcnn=None, show=True):
    global tokenizer
    tokenizer = tokenize
    start_token = tokenizer.word_index['<start>']
    end_token = tokenizer.word_index['<end>']
    decoder_input = [start_token]
    decoder_input = np.repeat(decoder_input, repeats=image.shape[0])
    output = tf.cast(tf.expand_dims(decoder_input, 1), dtype=tf.int32)  # tokens

    for i in range(40):
        dec_mask = create_masks_decoder(output)
        predictions, attention_weights = transformer(image, output, False, dec_mask,img_rcnn=img_rcnn)
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if tf.reduce_all(tf.math.equal(predicted_id, end_token)):
            break
        l = list(map(append_to_list, predicted_id.numpy(), names.numpy()))
        output = tf.concat([output, predicted_id], axis=-1)

    d_real_cap = []
    for i in cap_real.numpy()[0]:
        if i == 0:
            break
        d_real_cap.append(tokenizer.index_word[i])

    r_cap = " ".join([f" {i}" for i in d_real_cap])
    f_cap = " ".join([f" {i}" for i in l[0]])
    print(f'\nReal caption: {r_cap}')
    print(f'Gen  caption: {f_cap}')
    print('\n\n')

    if show:
        plt.figure()
        img = np.array(Image.open(names.numpy()[0].decode('utf-8')))
        plt.imshow(img)
        plt.show()

    return f_cap, r_cap, names
