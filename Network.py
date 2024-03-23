'''This file contains the code for the Generator and Discriminator models. 
The Generator model is used to generate the super-resolution images and the Discriminator model is used to classify the images as real or fake. 
The Generator model is based on the ResNet architecture and the Discriminator model is based on the VGG architecture.'''
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Conv2D
from keras.models import Model
from keras.layers import LeakyReLU, PReLU
from keras.layers import add

'''res_block_gen and up_sampling_block are the two functions which are used to create the generator model.'''
def res_block_gen(model, kernal_size, filters, strides):
    
    gen = model
    
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
        
    model = add([gen, model])
    
    return model
    
'''up_sampling_block is used to perform the upsampling of the low-resolution images.'''
def up_sampling_block(model, kernal_size, filters, strides):
    
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = UpSampling2D(size=2)(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    return model

'''discriminator_block is used to create the discriminator model.'''
def discriminator_block(model, filters, kernel_size, strides):
    
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    return model
'''The Generator class is used to create the generator model for super-resolution image synthesis.'''
class Generator(object):
    """
    The Generator class represents a generator model for super-resolution image synthesis.
    """

    def __init__(self, noise_shape):
        """
        Initializes a new instance of the Generator class.

        Args:
            noise_shape (tuple): The shape of the input noise tensor.
        """
        self.noise_shape = noise_shape

    def generator(self):
        """
        Generates the super-resolution image.

        Returns:
            keras.models.Model: The generator model.
        """
        gen_input = Input(shape=self.noise_shape)
        
        model = Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(gen_input)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
        
        gen_model = model
        
        '''Using 16 Residual Blocks'''
        for _ in range(16):
            model = res_block_gen(model, 3, 64, 1)
        
        model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(model)
        model = BatchNormalization(momentum=0.5)(model)
        model = add([gen_model, model])
        
        '''Using 2 UpSampling Blocks'''
        for _ in range(2):
            model = up_sampling_block(model, 3, 256, 1)
        
        model = Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(model)
        model = Activation('tanh')(model)
       
        generator_model = Model(inputs=gen_input, outputs=model)
        
        return generator_model

'''The Discriminator class is used to create the discriminator model for classifying the images as real or fake.'''
class Discriminator(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape
    
    def discriminator(self):
        
        dis_input = Input(shape = self.image_shape)
        
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(dis_input)
        model = LeakyReLU(alpha = 0.2)(model)
        
        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)
        
        model = Flatten()(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha = 0.2)(model)
       
        model = Dense(1)(model)
        model = Activation('sigmoid')(model) 
        
        discriminator_model = Model(inputs = dis_input, outputs = model)
        
        return discriminator_model
