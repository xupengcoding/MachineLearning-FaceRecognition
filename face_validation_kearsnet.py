import numpy as np
from keras.models import Graph

np.random.seed(123456)

import os
import pickle as pkl
import matplotlib as plt
import pickle
from keras.utils import np_utils

from  keras.layers.convolutional import  Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
import keras.optimizers

def load_data_xy(file_name):
    datas = []
    labels = []
    f = open(file_name, 'rb')
    x,y = pickle.load(f)
    datas.append(y)
    labels.append(x)
    combine_d = np.vstack(datas)
    combine_l = np.hstack(labels)
    return combine_l, combine_d



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
    keras_model.add_node(layer=Dense(200, activation='relu'),name='dense3',
                         inputs=['dense1', 'dense2'], merge_mode='concat')

    #sotmax
    keras_model.add_node(layer=Dense(2, activation='softmax'), name='dense4', input='dense3')
    keras_model.add_output('output', input='dense4')
    #add soft max  or min_max or maxpooling
    keras_model.compile('adadelta', {'output': 'categorical_crossentropy'},metrics=['accuracy'])
    return keras_model

    #history = keras_model.fit({'input1':X_train, 'input2':X2_train, 'output':y_train}, nb_epoch=10)
    #predictions = graph.predict({'input1':X_test, 'input2':X2_test}) # {'output':...}

train1_label_vec, train1_data_vec = load_data_xy("fr_train_data_feature/0.pkl")
train2_label_vec, train2_data_vec = load_data_xy("fr_train_data_feature/1.pkl")
train_label_vec = np.zeros(1000, dtype='int32')
train_label_vec = np_utils.to_categorical(train_label_vec, 2)

model = face_valid_net()
history = model.fit({'input1':train1_data_vec, 'input2':train2_data_vec, 'output':train_label_vec},validation_split = 0.3, nb_epoch=10)
score = model.evaluate({'input1':train1_data_vec, 'input2':train2_data_vec, 'output':train_label_vec})
moedel1 = face_valid_net()