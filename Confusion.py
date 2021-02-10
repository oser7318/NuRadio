import numpy as np 
import matplotlib as plt


import os
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True) 

from tensorflow import keras
from generator import TrainDataset, ValDataset, list_of_file_ids_val, n_events_per_file, n_files_train, n_files_val, batch_size, load_file
from sklearn.metrics import confusion_matrix



#Load data file

i_file = int(list_of_file_ids_val)

valdata, val_label_onehot = ValDataset._generator(0)

#Print some labels
for k in range(5):
    print(val_label_onehot(k))


#Recreate validation data
n_batches_per_file=n_events_per_file//batch_size

#dataset_val = tf.data.Dataset.range(n_files_val).prefetch(n_batches_per_file * 10).interleave(
#        ValDataset,
#        cycle_length=2,
#        num_parallel_calls=2,
#        deterministic=False)

#Load saved model
#model=keras.models.load_model('/mnt/md0/analysis/flavor/01/saved_models/T01/model_best.h5')

#Let model make predictions on validation dataset
#category_predictions = model.predict(dataset_val, batch_size=batch_size)

#print('Raw prediction values:')

#for i in category_predictions:
#    print(i)

#print('np.argmax on prediction values:')
#category_predictions = np.argmax(category_predictions, axis=1)
