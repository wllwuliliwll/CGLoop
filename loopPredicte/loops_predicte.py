# -*- coding: utf-8 -*-
import os
import re
import joblib
import argparse
import math
import pysam
import time
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
from os import listdir
from os.path import isfile, join
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from collections import Counter
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import StandardScaler 
from tensorflow.keras.layers import Input, Dense, Conv2D, DepthwiseConv2D, BatchNormalization, Dropout, GlobalAveragePooling2D, Reshape, multiply, add, Activation
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, average_precision_score,precision_recall_curve, auc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.random.set_seed(123)
def channel_attention(input_feature, ratio=8):

    #channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[-1]
    filters = max(1, int(channel//ratio))
    shared_layer_one = tf.keras.layers.Dense(filters,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = tf.keras.layers.Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(input_feature)    
    avg_pool = tf.keras.layers.Reshape((1,1,channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = tf.keras.layers.GlobalMaxPooling2D()(input_feature)
    max_pool = tf.keras.layers.Reshape((1,1,channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
   

    cbam_feature = tf.keras.layers.Add()([avg_pool,max_pool])
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)


    return multiply([input_feature, cbam_feature])
def spatial_attention(input_feature,kernel_siz):
    kernel_size = kernel_siz

    channel = input_feature.shape[-1]
    cbam_feature = input_feature

    avg_pool = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
  
    max_pool = tf.keras.layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    
    concat = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])

    cbam_feature = tf.keras.layers.Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='glorot_uniform',#'he_normal',
                    use_bias=False)(concat)	
    #assert cbam_feature._keras_shape[-1] == 1

    return multiply([input_feature, cbam_feature])
def cbam_block(cbam_feature, ratio=8, kernel_size=(2, 2)):
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature, kernel_size)
    return cbam_feature
def create_model():
    inputs = tf.keras.Input(shape=(21, 21, 1))
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3),activation='elu')(inputs)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = cbam_block(x)
    x = tf.keras.layers.SeparableConv2D(32, kernel_size=(3, 3),activation='elu')(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    cnn_model = tf.keras.Model(inputs, x)
    inputs = tf.keras.Input((None, 21, 21, 1))
    encoded_frames = tf.keras.layers.TimeDistributed(cnn_model)(inputs)
    encoded_sequence = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(encoded_frames)
    encoded_sequence = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True))(encoded_sequence)
    hidden_layer = tf.keras.layers.Dense(units=32, activation="relu")(encoded_sequence)
    hidden_layer = tf.keras.layers.Dropout(0.2)(hidden_layer)
    hidden_layer = tf.keras.layers.Dense(units=16, activation="relu")(hidden_layer)
    hidden_layer = tf.keras.layers.Dropout(0.2)(hidden_layer)
    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(hidden_layer)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.001),
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        metrics=[
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='prauc', curve='PR')          
        ]
      
    )
    
    return model


class load_test(Sequence):

    def __init__(self, x_y_set, batch_size):
        self.x_y_set = x_y_set
        self.x, self.y = self.x_y_set[:,:-1].reshape(-1,21,21),self.x_y_set[:,-1]
        self.batch_size = batch_size
        
    def __len__(self):
        return math.floor(len(self.x) / self.batch_size)
    def on_epoch_end(self):
        np.random.shuffle(self.x_y_set) 
        
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = batch_x.reshape(-1,self.batch_size,441,1)
        batch_y = batch_y.reshape(-1,self.batch_size)
        return np.array(batch_x), np.array(batch_y)

class WarmupExponentialDecay(Callback):
    def __init__(self,lr_base=0.0001,lr_min=0.0,decay=0,warmup_epochs=0):
        self.num_passed_batchs = 0  
        self.warmup_epochs=warmup_epochs  
        self.lr=lr_base #learning_rate_base
        self.lr_min=lr_min 
        self.decay=decay  
        self.steps_per_epoch=0 
    def on_batch_begin(self, batch, logs=None):
        if self.steps_per_epoch==0:
            if self.params['steps'] == None:
                self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
            else:
                self.steps_per_epoch = self.params['steps']
        if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:
            K.set_value(self.model.optimizer.lr,
                        self.lr*(self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs)
        else:
            K.set_value(self.model.optimizer.lr,
                        self.lr*((1-self.decay)**(self.num_passed_batchs-self.steps_per_epoch*self.warmup_epochs)))
        self.num_passed_batchs += 1
from keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss',  
                              factor=0.5,          
                              patience=3,          
                              min_lr=1e-7,         
                              verbose=1)
class PredictionDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.x = self.data.reshape(-1, 21, 21)  

    def __len__(self):
        return math.floor(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = batch_x.reshape(-1, self.batch_size, 441, 1)
        return np.array(batch_x)

def main(inputfile, outputfile, res):
    matrix_size = 21
    filepath = inputfile
    filename = filepath.split("/")[-1]  
    match = re.search(r'chr(\d+)', filename)

    if match:
        chr_name = 'chr' + match.group(1)
    else:
        print("chrname don't find!")
        return
    print(chr_name)
    test_data = np.load(filepath)
    location = test_data[:,:2]
    print(test_data.shape)
    data, infy = test_data[:, 2:], test_data[:, 222]
    scaler_filename = "../Model/scaler.pkl"
    scaler = joblib.load(scaler_filename)
    data1 = scaler.transform(data)
    batch_size = 50
    preprocessed_data = PredictionDataGenerator(data1, batch_size)
    epo = math.floor(len(data1) / 50)
    model = create_model()
    model.load_weights("../Model/F1score_model.h5")
    predictions = model.predict(preprocessed_data, batch_size=50)
    predictions = predictions.flatten()
    location = location[:50*epo]
    infy = infy[:50*epo]
    location10 = (location[:,0] - 1) * int(res)
    location11 = location[:,0] * int(res)
    location20 = (location[:,1] - 1) * int(res)
    location21 = location[:,1] * int(res)
    location10 = location10.astype(int)
    location11 = location11.astype(int)
    location20 = location20.astype(int)
    location21 = location21.astype(int)
    chrname = chr_name
    chrname_column = np.full((len(location10), 1), chrname)
    combined_data = np.column_stack((chrname_column, location10, location11, chrname_column, location20, location21, predictions.reshape(-1, 1), infy))
    np.savetxt(outputfile, combined_data, delimiter='\t', fmt='%s')
    current_time = datetime.now()
    print("Current time:", current_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction on Hi-C data")
    parser.add_argument("-i", "--inputfile", type=str, required=True, help="Path to the input .npy file")
    parser.add_argument("-o", "--outputfile", type=str, required=True, help="Path to the output .bedpe file")
    parser.add_argument("-r", "--res", type=str, required=True, help="Resolution for the location calculation")

    args = parser.parse_args()
    main(args.inputfile, args.outputfile, args.res)
