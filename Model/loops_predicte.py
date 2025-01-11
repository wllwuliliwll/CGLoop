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
from model import create_model
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import StandardScaler 
from tensorflow.keras.layers import Input, Dense, Conv2D, DepthwiseConv2D, BatchNormalization, Dropout, GlobalAveragePooling2D, Reshape, multiply, add, Activation
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, average_precision_score,precision_recall_curve, auc
tf.random.set_seed(123)
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
    scaler_filename = "./scaler.pkl"
    scaler = joblib.load(scaler_filename)
    data1 = scaler.transform(data)
    batch_size = 50
    preprocessed_data = PredictionDataGenerator(data1, batch_size)
    epo = math.floor(len(data1) / 50)
    model = create_model()
    model.load_weights("./F1score_model.h5")
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
