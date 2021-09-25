# from tensorflow.keras import layers
# from tensorflow.keras import Model
#
#
# def define_discriminator():
#     word_vector_dim = 300
#     dropout_prob = 0.4
#
#     in_label = layers.Input(shape=(300,))
#
#     n_nodes = 3 * 64 * 64
#     li = layers.Dense(n_nodes)(in_label)
#     li = layers.Reshape((64, 64, 3))(li)
#
#     dis_input = layers.Input(shape=(64, 64, 3))
#
#     merge = layers.Concatenate()([dis_input, li])
#
#     discriminator = layers.Conv2D(
#         filters=64, kernel_size=(3, 3), padding="same")(merge)
#     discriminator = layers.LeakyReLU(0.2)(discriminator)
#     discriminator = layers.GaussianNoise(0.2)(discriminator)
#
#     discriminator = layers.Conv2D(filters=64, kernel_size=(
#         3, 3), strides=(2, 2), padding="same")(discriminator)
#     discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
#     discriminator = layers.LeakyReLU()(discriminator)
#
#     discriminator = layers.Conv2D(filters=128, kernel_size=(
#         3, 3), padding="same")(discriminator)
#     discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
#     discriminator = layers.LeakyReLU(0.2)(discriminator)
#
#     discriminator = layers.Conv2D(filters=128, kernel_size=(
#         3, 3), strides=(2, 2), padding="same")(discriminator)
#     discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
#     discriminator = layers.LeakyReLU(0.2)(discriminator)
#
#     discriminator = layers.Conv2D(filters=256, kernel_size=(
#         3, 3), padding="same")(discriminator)
#     discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
#     discriminator = layers.LeakyReLU(0.2)(discriminator)
#
#     discriminator = layers.Conv2D(filters=256, kernel_size=(
#         3, 3), strides=(2, 2), padding="same")(discriminator)
#     discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
#     discriminator = layers.LeakyReLU(0.2)(discriminator)
#
#     discriminator = layers.Conv2D(filters=512, kernel_size=(
#         3, 3), padding="same")(discriminator)
#     discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
#     discriminator = layers.LeakyReLU(0.2)(discriminator)
#
#     discriminator = layers.Flatten()(discriminator)
#
#     discriminator = layers.Dense(1024)(discriminator)
#
#     discriminator = layers.LeakyReLU(0.2)(discriminator)
#
#     discriminator = layers.Dense(1)(discriminator)
#
#     discriminator_model = Model(
#         inputs=[dis_input, in_label], outputs=discriminator)
#
#     return discriminator_model
