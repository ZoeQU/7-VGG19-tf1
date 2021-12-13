# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from keras import Model, Sequential
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.preprocessing import image
from keras.utils import plot_model

# WEIGHTHS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTHS_PATH = 'vgg19_weights_tf_dim_ordering_tf_kernels.h5'

def VGG19(num_classes):
    image_input = Input(shape=(224, 224, 3))

    # (224, 224, 3) -> (112, 112, 64)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(image_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # (112, 112, 64) -> (56, 56, 128)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # (56, 56, 128) -> (28, 28, 256)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # (28, 28, 256) -> (14, 14, 512)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # (14, 14, 512) -> (7, 7, 512)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # (7, 7, 512) -> 25088 -> 4096 -> 4096 -> num_classes
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(image_input, x, name='vgg19')
    return model


if __name__ == '__main__':
    model = VGG19(1000)
    plot_model(model, 'vgg19net.svg', show_shapes=True)
    weights_path = tf.keras.utils.get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTHS_PATH, cache_subdir='models')
    model.load_weights(weights_path)
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('input image shape:', x.shape)
    preds = model.predict(x)
    print('predicted:', decode_predictions(preds))