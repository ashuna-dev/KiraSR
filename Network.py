from keras.layers import Dense, Activation, BatchNormalization, UpSampling2D, Flatten, Input, Conv2D
from keras.models import Model
from keras.layers import LeakyReLU, PReLU, add

def res_block_gen(model, kernel_size, filters, strides):
    """
    Generates a residual block for a given model.

    Args:
        model (tensorflow.keras.Model): The input model.
        kernel_size (int): The size of the convolutional kernel.
        filters (int): The number of filters in the convolutional layers.
        strides (int): The stride size for the convolutional layers.

    Returns:
        tensorflow.keras.Model: The model with the residual block added.
    """
    gen = model
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = add([gen, model])
    return model

def up_sampling_block(model, kernel_size, filters, strides):
    
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = UpSampling2D(size=2)(model)
    model = LeakyReLU(alpha=0.2)(model)
    return model

def discriminator_block(model, filters, kernel_size, strides):
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = LeakyReLU(alpha=0.2)(model)
    return model

class Generator(object):
    def __init__(self, noise_shape):
        self.noise_shape = noise_shape

    def generator(self):
        gen_input = Input(shape=self.noise_shape)
        model = Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(gen_input)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
        gen_model = model

        for _ in range(16):
            model = res_block_gen(model, 3, 64, 1)

        model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(model)
        model = BatchNormalization(momentum=0.5)(model)
        model = add([gen_model, model])

        for _ in range(2):
            model = up_sampling_block(model, 3, 256, 1)

        model = Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(model)
        model = Activation('tanh')(model)

        generator_model = Model(inputs=gen_input, outputs=model)
        return generator_model

class Discriminator(object):
    """
    A class representing the discriminator network.

    Parameters:
    image_shape (tuple): The shape of the input image.

    Returns:
    keras.Model: The compiled discriminator model.
    """

    def __init__(self, image_shape):
        self.image_shape = image_shape

    def discriminator(self):
        """
        Creates the discriminator model.

        Returns:
        keras.Model: The compiled discriminator model.
        """
        dis_input = Input(shape=self.image_shape)
        model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(dis_input)
        model = LeakyReLU(alpha=0.2)(model)

        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)

        model = Flatten()(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)

        model = Dense(1)(model)
        model = Activation('sigmoid')(model)

        discriminator_model = Model(inputs=dis_input, outputs=model)
        return discriminator_model
