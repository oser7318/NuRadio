import numpy as np 
import matplotlib.pyplot as plt
import itertools


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

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig=plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.savefig(f'{title.replace(" ", "_")}.png')

#Recreate validation data
n_batches_per_file=n_events_per_file//batch_size

i_file = list_of_file_ids_val[0]

val_data, true_category = load_file(i_file)

true_category=np.argmax(true_category,axis=1)

comp_true_category = true_category[0:10000]
comp_predicted_category = category_predictions[0:10000]

#Load saved model
model=keras.models.load_model('/mnt/md0/analysis/flavor/01/saved_models/T01/model_best.h5')

#Let model make predictions on validation dataset
category_predictions = model.predict(val_data, batch_size=batch_size)
category_predictions = np.argmax(category_predictions, axis=1)

#Extract the true category values from the validation data set

#Create confusion matrix using scikit learn built in confusion matrix function
cm = confusion_matrix(y_true=comp_true_category, y_pred=comp_predicted_category)
cm_plot_labels = ['Hadronic only', 'EM + Hadronic']

#Plot the confusion matrix
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion matrix for T01')
