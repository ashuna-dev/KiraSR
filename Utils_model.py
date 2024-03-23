from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.optimizers import Adam

import keras.backend as K

''' This class is used to calculate the VGG loss between the high-resolution and the generated high-resolution images.'''
class VGG_LOSS(object):
    def __init__(self, image_shape):
        self.image_shape = image_shape

    def vgg_loss(self, y_true, y_pred):
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False

        ''' Make the layers of the VGG19 network as non-trainable.'''
        for layer in vgg19.layers:
            layer.trainable = False

        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False

        return K.mean(K.square(model(y_true) - model(y_pred)))

''' This function is used to get the optimizer for the model.'''
def get_optimizer():
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    return adam
