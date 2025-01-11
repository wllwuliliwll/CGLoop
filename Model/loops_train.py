# train.py
import math
import os
import pickle
import joblib
import tensorflow as tf
import numpy as np
import argparse
from tensorflow import keras
from tensorflow.keras import layers
from os import listdir
from os.path import isfile, join
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from collections import Counter
from model import create_model 
from sklearn.metrics import average_precision_score
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler 
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Conv2D, DepthwiseConv2D, BatchNormalization, Dropout, GlobalAveragePooling2D, Reshape, multiply, add, Activation
from sklearn.metrics import precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')
tf.random.set_seed(123)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os
import GPUtil
import warnings
warnings.filterwarnings("ignore")
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.set_visible_devices(physical_devices[2], 'GPU') # 验证是否选中了 GPU 0 
print("Using GPU: ", physical_devices[2].name)
tf.config.experimental.set_memory_growth(physical_devices[2], True)
def get_gpu_memory_usage_by_pid():
    command = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        pid_str, memory_used = line.split(", ")
        pid_in_line = int(pid_str)
        memory_used_in_mb = int(memory_used)
        if pid_in_line == current_pid:
            return memory_used_in_mb    
    return None  
# data
class load_test(Sequence):
    def __init__(self, x_y_set, batch_size):
        self.x_y_set = x_y_set
        self.x, self.y = self.x_y_set[:, :-1].reshape(-1,21,21), self.x_y_set[:, -1]
        self.batch_size = batch_size
        
    def __len__(self):
        return math.floor(len(self.x) / self.batch_size)
    
    def on_epoch_end(self):
        np.random.shuffle(self.x_y_set) 
        
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = batch_x.reshape(-1, self.batch_size, 441, 1)
        batch_y = batch_y.reshape(-1, self.batch_size)
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
# main
def main(trainfile,valfile):
    train_data_before = np.load(trainfile)
    val_data_before = np.load(valfile)# Separate features and labels
    X_train, y_train = train_data_before[:, :-1], train_data_before[:, -1]
    X_val, y_val = val_data_before[:, :-1], val_data_before[:, -1]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    with open(f'./Model/scaler.pkl', 'wb') as f:
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
        print(f"Epoch {epoch}")
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
            model.save_weights('./Model/F1score_model.h5')
        print('The best f1: ', bestf1)
        if(bestprauc<val_prauc):        
            bestprauc = val_prauc
            print('New bestprauc: ', bestprauc)
            model.save_weights('./Model/Prauc_model.h5')
        print('The bestprauc: ', bestprauc)
        gpu_memory = get_gpu_memory_usage_by_pid() 
        print(f"Epoch {epoch} GPU memory used: {gpu_memory_usage:.2f} MB")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run train on Hi-C data")
    parser.add_argument("-t", "--train_file", type=str, required=True, help="Path to the train .npy file")
    parser.add_argument("-v", "--val_file", type=str, required=True, help="Path to the val .npy file")
    args = parser.parse_args()
    main(args.train_file, args.val_file)
