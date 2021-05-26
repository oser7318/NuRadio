import matplotlib.pyplot as plt
import numpy as np

from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True) 

from tensorflow import keras
from generator import load_file, list_of_file_ids_test, TestDataset


model=keras.models.load_model('/mnt/md0/oericsson/NuRadio/saved_models/FINN04/model_best.h5')

i_file= list_of_file_ids_test[0]

data, labels = load_file(i_file, noise=True, em=True)

max_LPDA = np.max(np.max(np.abs(data[:, :, 0:4]), axis=1), axis=1)
SNR = max_LPDA[:,0]/10 

predictions = model.predict(data)

#----SNR Bins----
#4 events with SNR<1. min(SNR) = 0.9250, max(SNR)=271.1495. 98841 events in interval 1<SNR<5.

SNR_bins = np.linspace(1, 10, num=37 endpoint=True) #np.array([min(SNR), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, max(SNR)])

#ind = np.where((SNR>=min(SNR)) & (SNR<1.5))
accuracy = []
events_per_bin = []
for i in range(len(SNR_bins)-1):

    #indeces = np.where((shower_energy_em > energies[i]) & (shower_energy_em < energies[i+1])) #Finds indeces of events where shower_energy_em is in the specified interval 
    indeces = np.where((SNR > SNR_bins[i]) & (SNR < SNR_bins[i+1])) #Finds indeces of events where nu_energy is in the specified interval 
    print(f'indeces shape: {indeces[0].shape}')
    reduced_predictions = np.argmax(predictions[indeces], axis=1) #Array consisting of argmaxed predicted labels for only those events where nu_energy/shower_energy_em is within the interval. i.e. 1 for e cc and 0 for the rest. 
    reduced_labels = np.argmax(labels[0][indeces], axis=1)

    mask_pred = reduced_predictions == reduced_labels

    correct_predictions = np.sum(mask_pred)
    
    #OBS THIS ONLY WORKS FOR DATASETS OF ONLY e CC EVETNS! i.e. all events are [0,1] => argmax gives 1.
    #correct_predictions = np.count_nonzero( np.argmax(reduced_predictions, axis=1) == 1 )

    acc = 100 * correct_predictions/len(reduced_predictions)
    accuracy.append(acc)
    events_current_bin = len(indeces)
    events_per_bin.append(events_current_bin)
    print(f'SNR interval: ({SNR_bins[i]},{SNR_bins[i+1]})\t Accuracy: {acc}%')

SNR_bin_points = [ (SNR_bins[j]+SNR_bins[j+1])/2 for j in range(len(SNR_bins)-1) ]
#SNR_bin_points.append(15+16/2)

# min_err = np.ones(15)*0.5
# pos_err = np.ones(15)*0.5
# pos_err[-1] = 0

plt.errorbar(SNR_bin_points, accuracy, xerr=((SNR_bins[0]+SNR_bins[1])/2), fmt='bo', capsize=5, elinewidth=1, markeredgewidth=1)
plt.ylim(0,100)
plt.xlabel('SNR')
plt.ylabel('Percentage $\u03BD_e$ CC events classified as such')
plt.title('Dependence of accuracy on signal-to-noise ratio (SNR)')

plt.tight_layout()
plt.savefig('SNR_EM.png')

plt.clf()