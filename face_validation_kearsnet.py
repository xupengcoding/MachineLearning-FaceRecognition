import numpy as np
from keras.models import Graph

np.random.seed(123456)

import os
import pickle as pkl
import matplotlib as plt

from  keras.layers.convolutional import  Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
import keras.optimizers

def face_valid_net():
    #a graph model
    keras_model = Graph()
    #input 2268 dim
    #two input layer
    keras_model.add_input(name="input1", input_shape=(2622,))
    keras_model.add_input(name="input2", input_shape=(2622,))

    #two dense layer
    keras_model.add_node(layer=Dense(1000, activation='relu'), name='dense1', input='input1')
    keras_model.add_node(layer=Dense(1000, activation='relu'), name='dense2', input='input2')
    #concat layer
    keras_model.add_node(layer=Dense(200, activation='rely'),name='dense3',
                         inputs=['dense1', 'dense2'], merge_mode='concat')

    #sotmax
    keras_model.add_node(layer=Dense(2, activation='softmax'), name='dens4', input='dense3')
    keras_model.add_output('output1', input='dense4')
    #add soft max  or min_max or maxpooling
    keras_model.compile('adadelta', {'output', 'categorical_crossentropy'})

    history = graph.fit({'input1':X_train, 'input2':X2_train, 'output':y_train}, nb_epoch=10)  
predictions = graph.predict({'input1':X_test, 'input2':X2_test}) # {'output':...} 