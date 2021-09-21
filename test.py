# from train import *
from inference import evaluate
from tempo import *


# def load_pre_image(image_path):
#     img = tf.io.read_file(image_path)
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.image.resize(img, (64, 64))
#     return img
#
#
# def map_func(img_name, cap):
#     img_tensor = np.load(img_name.decode('utf-8') + '.npy')
#     image = load_pre_image(img_name.decode('utf-8'))
#     return img_tensor, cap, img_name, image
#
#
# i_data = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))
# i_data = i_data.map(
#     lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32, tf.string, tf.float32]),
#     num_parallel_calls=tf.data.experimental.AUTOTUNE)
# i_data = i_data.shuffle(BUFFER_SIZE).batch(1)
# i_data = i_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# for idx, (img_tensor, cap, img_name, image) in enumerate(i_dataset):
#     evaluate(img_tensor, img_name, cap, tokenizer, transformer)

# for idx, (img_tensor, cap, img_name, image) in enumerate(i_data):
#     evaluate(img_tensor, img_name, cap, tokenizer, transformer)
