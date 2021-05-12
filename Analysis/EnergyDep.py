import matplotlib.pyplot as plt
import numpy as np
import itertools
from math import log10

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

print("Loading model...")
model=keras.models.load_model('/mnt/md0/oericsson/NuRadio/saved_models/NuFlavorCNN8.21File/model_best.h5')

i_file= list_of_file_ids_test[0]

# print("Done!\n Loading labels...")

# datapath = "/mnt/ssd2/data/SouthPole/single_surface_4LPDA_PA_15m_RNOG_fullsim.json/ARZ2020_emhad_noise.yaml/G03generate_events_full_surface_sim/LPDA_2of4_100Hz/4LPDA_1dipole_fullband/em_had_separately"
# labels = np.load(os.path.join(datapath, f"labels_emhad_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_{i_file:04d}.npy"), allow_pickle=True)




#print("Done!")

print(f'Loading file {i_file}...')

test_data, test_labels = load_file(i_file, noise=True, em=True)
#shower_energy_em = test_labels[1] 
nu_energy = test_labels[2]
#print(f'shower energy em shape: {shower_energy_em.shape}')

print("Done!")

print("Making predictons...")

predictions = model.predict(test_data)
print(f'predicitons shape: {predictions.shape}')
print("Done!")

#for-loop iterating over energy intervals and saving the accuracies to a list: 
accuracy = []
energies = np.logspace(17, 19, 17) #For file 19: Only 41 events with shower_energy_em < 1e15, 432 events < 1e16. 

for i in range(len(energies)-1):

    indeces = np.where((nu_energy > energies[i]) & (nu_energy < energies[i+1])) #Finds indeces of events where shower_energy_em is in the specified interval 
    print(f'indeces shape: {indeces[0].shape}')
    reduced_predictions = np.argmax(predictions[indeces], axis=1) #Array consisting of argmaxed predicted labels for only those events where shower_energy_em is within the interval. i.e. 1 for e cc and 0 for the rest. 
    reduced_test_labels = np.argmax(test_labels[0][indeces], axis=1)

    mask_pred = reduced_predictions == reduced_test_labels

    correct_predictions = np.sum(mask_pred)
    
    #OBS THIS ONLY WORKS FOR DATASETS OF ONLY e CC EVETNS! i.e. all events are [0,1] => argmax gives 1.
    #correct_predictions = np.count_nonzero( np.argmax(reduced_predictions, axis=1) == 1 )

    acc = 100 * correct_predictions/len(reduced_predictions)
    accuracy.append(acc)

    print(f'Energy interval: ({energies[i]},{energies[i+1]})\t Accuracy: {acc}%')

log_energies = [log10(energy) for energy in energies]

x = [ (log_energies[j]+log_energies[j+1])/2 for j in range(len(log_energies)-1) ]

plt.errorbar(x, accuracy, xerr=((log_energies[1]-log_energies[0])/2), fmt='bo', capsize=5, elinewidth=1, markeredgewidth=1)
plt.xlim(16.5,19.5)
plt.ylim(0,100)
plt.xlabel('log($E_{EM}$)')
plt.ylabel('Percentage $\u03BD_e$ CC events classified as such')
plt.title('Dependence of accuracy on EM shower energy')

plt.tight_layout()
plt.savefig('EnergyDep.png')

plt.clf()