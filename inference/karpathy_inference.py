import json
import warnings
import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")
from sklearn.utils import shuffle

results = {}
karpathy_test = './Dataset/COCO/splits/finalkarpathysplit_test.json'

f = open(karpathy_test)
dict1 = json.load(f)
dict2 = dict()

for keys in dict1:
    dict2[keys['image_id']] = keys['dir']


def append_to_list(id, name):
    word = tokenizer.index_word[int(id[0])]
    if name in results.keys():
        if results[name][len(results[name]) - 1] != '<end>':
            results[name].append(word)
    else:
        results[name] = [word]

    return results[name]


def i_map_func(img_name):
    a = dict2.get(img_name)
    a = a.split('.')
    img_tensor = np.load('./Dataset/COCO' + a[1] + '.jpg.npy')
    return img_tensor, img_name


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


def evaluate(image, names, tokenize, transformer, show=True):
    global tokenizer
    tokenizer = tokenize
    start_token = tokenizer.word_index['<start>']
    end_token = tokenizer.word_index['<end>']
    decoder_input = [start_token]
    decoder_input = np.repeat(decoder_input, repeats=image.shape[0])
    output = tf.cast(tf.expand_dims(decoder_input, 1), dtype=tf.int32)  # tokens

    for i in range(40):
        dec_mask = create_masks_decoder(output)
        predictions, attention_weights = transformer(image, output, False, dec_mask)
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if tf.reduce_all(tf.math.equal(predicted_id, end_token)):
            break
        l = list(map(append_to_list, predicted_id.numpy(), names.numpy()))
        output = tf.concat([output, predicted_id], axis=-1)


def karpathy_inference(tokenizer, transformer):
    l = (dict2.keys())
    l = list(set(l))

    i_data = tf.data.Dataset.from_tensor_slices(l)
    i_data = i_data.map(lambda item1: tf.numpy_function(i_map_func, [item1], [tf.float32, tf.int32]),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    i_data = i_data.batch(64)
    i_data = i_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    for batch_idx, (image, names) in enumerate(i_data):
        print(f'Karpathy split inference : {batch_idx}')
        evaluate(image, names, tokenizer, transformer)

    finallist = []
    for i in results.keys():
        imagecap = results.get(i)
        result_join = ' '.join(imagecap)
        result_join = result_join[:(len(result_join) - 6)]
        finallist.append({'image_id': int(i), 'caption': result_join})

    jsonString = json.dumps(finallist)
    jsonFile = open("./captions_val2014_result_results.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()
