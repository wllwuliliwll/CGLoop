import pysam
import numpy as np
import pandas as pd
import time
import math
import os
import pickle
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from os import listdir
from os.path import isfile, join
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from collections import Counter
from sklearn.metrics import average_precision_score
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler 
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Conv2D, DepthwiseConv2D, BatchNormalization, Dropout, GlobalAveragePooling2D, Reshape, multiply, add, Activation
from sklearn.metrics import precision_recall_curve, auc
tf.random.set_seed(123)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def channel_attention(input_feature, ratio=8):
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
    #assert concat._keras_shape[-1] == 2
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
        #loss=weighted_binary_crossentropy_with_label_smoothing,
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
        #self.x = scaler.fit_transform(self.x.astype(np.float32).reshape(-1,1)).reshape(-1,21,21)
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
        self.lr=lr_base
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

reduce_lr = ReduceLROnPlateau(monitor='val_loss',  
                              factor=0.5,          
                              patience=3,         
                              min_lr=1e-7,  
                              verbose=1)

del main(trainfile,valfile):
    train_data_before = np.load(trainfile)
    val_data_before = np.load(valfile)# Separate features and labels
    X_train, y_train = train_data_before[:, :-1], train_data_before[:, -1]
    X_val, y_val = val_data_before[:, :-1], val_data_before[:, -1]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    with open(f'../Model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    # Concatenate the scaled features and labels back together
    train_data = np.column_stack((X_train_scaled, y_train))
    val_data = np.column_stack((X_val_scaled, y_val))
    print(train_data.shape)
    print(val_data.shape)
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    #np.random.shuffle(data)
    loss_list = []
    accuracy_list = []
    val_list = []
    val_loss_list = []
    val_accuracy_list = []
    f1_list = []
    val_prauc_list = []
    bestf1 = 0.0
    bestprauc=0.0
    epoch=20
    batchsize=50
    train = load_test(train_data,batchsize)
    validation1 = load_test(val_data,batchsize)
    model = create_model()
    for epoch in range(1, 50):
        print(epoch)
        warmup_lr = WarmupExponentialDecay(lr_base=0.0001, decay=0.00001, warmup_epochs=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
        history = model.fit(train, validation_data=validation1,callbacks=[warmup_lr, reduce_lr])    
        loss_list.append(history.history["loss"])
        accuracy_list.append(history.history["accuracy"])   
        val_loss_list.append(history.history["val_loss"])
        val_accuracy_list.append(history.history["val_accuracy"]) 
        val_prauc_list.append(history.history["val_prauc"])     
        results = model.evaluate(validation1)
        recall_index = model.metrics_names.index('recall')
        precision_index = model.metrics_names.index('precision')
        prauc_index = model.metrics_names.index('prauc')
        val_recall = results[recall_index]
        val_precision = results[precision_index]
        val_prauc = results[prauc_index]
        f1 = 2 * (val_recall *val_precision) / (val_recall + val_precision + 1e-10)
        f1_list.append(f1)
        val_prauc_list.append(val_prauc)
        if(bestf1<f1):        
            bestf1 = f1
            print('New best f1: ', bestf1)
            model.save_weights(f'../Model/F1score_model.h5')
        print('The best f1: ', bestf1)
        if(bestprauc<val_prauc):        
            bestprauc = val_prauc
            print('New bestprauc: ', bestprauc)
            model.save_weights(f'../Model/Prauc_model.h5')
        print('The bestprauc: ', bestprauc)   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run train on Hi-C data")
    parser.add_argument("-t", "--train_file", type=str, required=True, help="Path to the train .npy file")
    parser.add_argument("-v", "--val_file", type=str, required=True, help="Path to the val .npy file")
    #parser.add_argument("-o", "--output_file", type=str, required=True, help="Path to save file")
    args = parser.parse_args()
    main(args.inputfile, args.outputfile)