import matplotlib.pyplot as plt
import numpy as np
import itertools


import os
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True) 

from tensorflow import keras
from generator import load_file, list_of_file_ids_test, batch_size


#Predict from test file --> Of files with EM shower energy in some interval, what was the accuracy? 

# print("Loading model...")
# model=keras.models.load_model('/mnt/md0/oericsson/NuRadio/saved_models/NuFlavorCNN4_2/model_best.h5')

i_file=list_of_file_ids_test[0]

print("Done!\n Loading labels...")

datapath = "/mnt/md0/data/SouthPole/single_surface_4LPDA_PA_15m_RNOG_fullsim.json/ARZ2020_emhad_noise.yaml/G03generate_events_full_surface_sim/LPDA_2of4_100Hz/4LPDA_1dipole_fullband/em_had_separately"
labels = np.load(os.path.join(datapath, f"labels_emhad_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_{i_file:04d}.npy"), allow_pickle=True)
shower_energy_em = np.array(labels.item()["shower_energy_em"])

print("Done!")

print(f'Loading file {i_file}...')

test_data, test_labels = load_file(i_file, noise=True, em=True)

print("Done!")

# print("Making predictons...")

# predictions = model.predict(test_data)

# print("Done!")
#Gives category predictions, data is 100% nu_e cc interaction
#Find indeces of energies in certain range

#Some fake predictions
predictions = np.array([[0,1]] * 100000)
print(f'Fake predictions:{predictions}')

indeces = np.where(shower_energy_em > 1e18) #Finds indeces of events where shower_energy_em is larger than 1e18 
print(indeces)
reduced_predictions = predictions[indeces] #Array consisting of predicted labels for only those events where shower_energy_em is larger than 1e18.


#print(f'Predictions:\n{predictions}')

correct_predictions = np.count_nonzero( np.argmax(reduced_predictions, axis=1) == 1 ) #counts the number of correct predictions (finds how many times the predicted label is [0,1] i.e. e cc interaction)

print(f'Number of correct predictions:{correct_predictions}\nNumber of predictions:{len(reduced_predictions)}')

accuracy = 100 * correct_predictions/len(reduced_predictions)

print(f'Accuracy above 1e18 energy is {accuracy}%')




