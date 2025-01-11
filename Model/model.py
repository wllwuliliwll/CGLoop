# model.py
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import scripts  
def create_model():
    inputs = tf.keras.Input(shape=(21, 21, 1))
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='elu')(inputs)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = scripts.cbam(x)
    x = tf.keras.layers.SeparableConv2D(32, kernel_size=(3, 3), activation='elu')(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    cnn_model = tf.keras.Model(inputs, x)    
    inputs = tf.keras.Input((None, 21, 21, 1))
    encoded_fea1 = tf.keras.layers.TimeDistributed(cnn_model)(inputs)
    encoded_fea2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(encoded_fea1)
    encoded_fea3 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True))(encoded_fea2)
    layer1 = tf.keras.layers.Dense(units=32, activation="relu")(encoded_fea3)
    layer2 = tf.keras.layers.Dropout(0.2)(layer1)
    layer3 = tf.keras.layers.Dense(units=16, activation="relu")(layer2)
    layer4 = tf.keras.layers.Dropout(0.2)(layer3)
    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(layer4)
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
