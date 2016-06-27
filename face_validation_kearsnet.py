import numpy as np
from keras.models import Graph

np.random.seed(123456)

import os
import sys

import pickle as pkl
import matplotlib as plt
import pickle
from keras.utils import np_utils
from  keras.layers.convolutional import  Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
import keras.optimizers
from keras.callbacks import ModelCheckpoint
from ml_file_pkg.pickle_file import load_data_xy
from ml_file_pkg.pickle_file import get_files


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


if __name__ == "__main__":
    train1_data_path = str(sys.argv[1])
    train2_data_path = str(sys.argv[2])
    #read train1_data
    train1_data_path_vec = get_files(train1_data_path)
    train1_data_vec, train1_label_vec = load_data_xy(train1_data_path_vec)
    #read train2_data
    train2_data_path_vec = get_files(train2_data_path)
    train2_data_vec, train2_label_vec = load_data_xy(train2_data_path_vec)

    #genereate output label
    train1_len = len(train1_label_vec)
    train2_len = len(train2_label_vec)
    assert train1_len == train2_len
    train_label_vec = np.zeros(train1_len, dtype='int32')
    for i in range(train1_len):
        if train1_label_vec[i] == train2_label_vec[i]:#the same person
            train_label_vec[i] = 1
        else:
            train_label_vec[i] = 0

    train_label_vec = np_utils.to_categorical(train_label_vec, 2)

    model = face_valid_net()
    model.summary()
    checkpointer = ModelCheckpoint(filepath= 'weights-{epoch:02d}-{val_loss:.2f}.hdf5',verbose=1, save_best_only=True)
    history = model.fit({'input1':train1_data_vec, 'input2':train2_data_vec, 'output':train_label_vec},validation_split = 0.1, nb_epoch=10, callbacks=[checkpointer])
    model.save_weights("validation_net_weitghts_last.hdf5", True)
    score = model.evaluate({'input1':train1_data_vec, 'input2':train2_data_vec, 'output':train_label_vec})
