#-------------------------------------------------------------------------
#------------------ PLACE HOLDER NAME ----------------------------------
#-------------------------------------------------------------------------


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
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.utils import plot_model

import pickle
import os
import matplotlib.pyplot as plt
import time
from generator import TrainDatasetEven, ValDatasetEven, n_events_per_file, n_files_train, n_files_val, batch_size, load_file

import wandb
from wandb.keras import WandbCallback

wandb.login()

hyperparam_default = dict(
    kernel_size = (1,5), #(1,10) originally
    poolsize = (1,3),
    lr = 1e-4,
)

#Initialize WandB with project name ("flavor-classification") and optionally with configurations.
run = wandb.init(config=hyperparam_default,project='flavor-classification')
config = wandb.config

filename = os.path.splitext(__file__)[0]
path = os.path.join('saved_models', filename)
if not os.path.exists(path):
    os.makedirs(path)


def conv_block(nfilters, nlayers=1, dropout=False, **kwargs):
    """ Add block of convolutional layers of specified number of filters, with or without dropout """

    for _ in range(nlayers):
        model.add(Conv2D(nfilters, config.kernel_size, padding="same", activation="relu", **kwargs))

    if dropout:
        model.add(Dropout(dropout))
    else: 
        model.add(MaxPooling2D(config.poolsize, padding="same"))


#-----------Define Model------------
#model = keras.models.load_model('/mnt/md0/oericsson/NuRadio/saved_models/NuFlavorCNN8/model_best.h5')
model = Sequential()

#convolutional layers
#Input layer
model.add(Conv2D(32, config.kernel_size, padding="same", input_shape=(5, 512, 1), activation="relu"))
conv_block(32, nlayers=3)
conv_block(64, nlayers=3)
conv_block(128, nlayers=3)
conv_block(256, nlayers=3)

#10 total conv2d layers

#Dense
model.add(Flatten())

model.add(Dropout(0.4))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(2, activation="softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=config.lr),
              metrics=["accuracy"])
model.summary()

#plot_model(model, to_file='4_2_architecture.png', show_shapes=True)

#model_json = model.to_json()
#with open(f'4_2_architecture.json', "w") as json_file:
#    json_file.write(model_json)

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

ES = EarlyStopping(monitor="val_accuracy", patience=3)
#reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, min_lr=1e-6)

history = model.fit(x=dataset_train, steps_per_epoch=steps_per_epoch, epochs=100,
          validation_data=dataset_val, callbacks=[checkpoint, csv_logger, ES, WandbCallback()]) #WandbCallbacks()
with open(os.path.join('saved_models', filename, 'history.pkl'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

