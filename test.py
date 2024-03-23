import argparse
import os
import numpy as np
from keras.models import load_model
import cv2
from skimage import io
from skimage.transform import resize
import Utils
import Utils_model
from Utils_model import VGG_LOSS

import matplotlib.pyplot as plt

''' This file is used to test the model output for the given input images.'''

image_shape = (96, 96, 3)

def test_model(input_hig_res, model, number_of_images, output_dir):
    x_test_lr, x_test_hr = Utils.load_test_data_for_model(input_hig_res, 'jpg', number_of_images)
    Utils.plot_test_generated_images_for_model(output_dir, model, x_test_hr, x_test_lr)

def test_model_for_lr_images(input_low_res, model, number_of_images, output_dir):
    x_test_lr = Utils.load_test_data(input_low_res, 'jpg', number_of_images)
    Utils.plot_test_generated_images(output_dir, model, x_test_lr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ihr', '--input_hig_res', action='store', dest='input_hig_res', default='./data/',
                        help='Path for input images High resolution')
    parser.add_argument('-ilr', '--input_low_res', action='store', dest='input_low_res', default='./data_lr/',
                        help='Path for input images Low resolution')
    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./output/',
                        help='Path for output images')
    parser.add_argument('-m', '--model_dir', action='store', dest='model_dir', default='./model/gen_model3000.h5',
                        help='Path for model')
    parser.add_argument('-n', '--number_of_images', action='store', dest='number_of_images', default=25,
                        help='Number of images', type=int)
    parser.add_argument('-t', '--test_type', action='store', dest='test_type', default='test_model',
                        help='Option to test model output or to test low resolution image')
    values = parser.parse_args()

    loss = VGG_LOSS(image_shape)
    model = load_model(values.model_dir, custom_objects={'vgg_loss': loss.vgg_loss})

    if values.test_type == 'test_model':
        test_model(values.input_hig_res, model, values.number_of_images, values.output_dir)
    elif values.test_type == 'test_lr_images':
        test_model_for_lr_images(values.input_low_res, model, values.number_of_images, values.output_dir)
    else:
        print("No such option")
