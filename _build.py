from black import Line
import tensorflow as tf
from tensorflow import keras

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


class Linear(keras.layers.Layer):
    def __init__(self, units=32) -> None:
        super().__init__()
        self.units = units

        # self.w = self.add_weight(
        #     shape=(input_shape[-1], self.units),
        #     initializer="random_normal",
        #     trainable=True,
        # )
        # self.b = self.add_weight(
        #     shape=(self.units,), initializer="random_normal", trainable=True
        # )

    def build(self, input_shape):
        self.layer = tf.keras.layers.Dense(self.units)

    # def build(self, input_shape):
    #     self.w = self.add_weight(
    #         shape=(input_shape[-1], self.units),
    #         initializer="random_normal",
    #         trainable=True,
    #     )
    #     self.b = self.add_weight(
    #         shape=(self.units,), initializer="random_normal", trainable=True
    #     )

    def call(self, inputs):
        return self.layer(inputs)


class MLPBlock(keras.Model):
    def __init__(self, input_shape) -> None:
        super().__init__()

    def build(self, input_shape):
        self.linear_1 = Linear()
        self.linear_2 = Linear()
        self.linear_3 = Linear(1)

    def call(self, inputs, y_hat):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)


mlp = MLPBlock((3, 64))
mlp.build((3, 64))
print(len(mlp.weights))
print(mlp.weights)
# y = mlp(tf.ones(shape=(3, 64)))  # The first call to the `mlp` will create the weights
# print(len(mlp.weights))
