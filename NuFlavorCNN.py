#-------------------------------------------------------------------------
#------------------ PLACE HOLDER NAME ----------------------------------
#-------------------------------------------------------------------------

import wandb
from wandb.keras import WandbCallback

import os
from gpuutils import GpuUtils

GpuUtils.allocate(gpu_count=1,
                  framework='keras')
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import numpy as np
import matplotlib
import tensorflow.keras.backend as K
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, Flatten, Dropout
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, AveragePooling1D, Input, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger

import pickle
import os
import matplotlib.pyplot as plt
import time
from generator import TrainDatasetNoiseless, ValDatasetNoiseless, n_events_per_file, n_files_train, n_files_val, batch_size, load_file

import wandb
from wandb.keras import WandbCallback

wandb.login()

# Initialize WandB with project name ("flavor-classification") and optionally with configurations.
run = wandb.init(project='flavor-classification')

filename = os.path.splitext(__file__)[0]
path = os.path.join('saved_models', filename)
if not os.path.exists(path):
    os.makedirs(path)


def conv_block(nlayers=1, nfilters=1, dropout=False, **kwargs):
    """ Add block of convolutional layers of specified number of filters, with or without dropout """

    for _ in range(nlayers):
        model.add(Conv2D(nfilters, (1,10), padding="same", kernel_initializer="he_normal", **kwargs))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

    if dropout:
        model.add(Dropout(dropout))
    else: 
        model.add(MaxPooling2D((1,2), padding="same") )


#-----------Define Model------------
model = Sequential()

#convolutional layers
conv_block(32, dropout=0.2, input_shape=(5, 512, 1))
conv_block(32)
conv_block(64, dropout=0.2)
conv_block(64)
conv_block(2, 128, dropout=0.2)
conv_block(128)
conv_block(2, 256, dropout=0.2)
conv_block(256)

#Dense
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(layers.BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(Dense(2))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-3),
              metrics=["accuracy"])
model.summary()

checkpoint = ModelCheckpoint(filepath=os.path.join('saved_models', filename, "model_best.h5"),
                                                   monitor='val_accuracy',
                             verbose=1,
                            save_best_only=True, mode='auto',
                            save_weights_only=False)
csv_logger = CSVLogger(os.path.join('saved_models', filename, "model_history_log.csv"), append=True)

steps_per_epoch = n_files_train * (n_events_per_file // batch_size)
n_batches_per_file = n_events_per_file // batch_size
print(f"steps_per_epoch {steps_per_epoch}, n_batches_per_file {n_batches_per_file}")

dataset_train = tf.data.Dataset.range(n_files_train).prefetch(n_batches_per_file * 10).interleave(
        TrainDatasetEven,
        cycle_length=2,
        num_parallel_calls=2,
        deterministic=False).repeat()

dataset_val = tf.data.Dataset.range(n_files_val).prefetch(n_batches_per_file * 10).interleave(
        ValDatasetEven,
        cycle_length=2,
        num_parallel_calls=2,
        deterministic=False)

history = model.fit(x=dataset_train, class_weight = d_class_weights, steps_per_epoch=steps_per_epoch, epochs=20,
          validation_data=dataset_val, callbacks=[checkpoint, csv_logger, WandbCallback()])
with open(os.path.join('saved_models', filename, 'history.pkl'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

