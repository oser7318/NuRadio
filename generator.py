import os
import numpy as np
import tensorflow as tf
import time

np.set_printoptions(precision=4)

#Category separated Noiseless data
#datapath = "/mnt/md0/data/SouthPole/single_surface_4LPDA_PA_15m_RNOG_fullsim.json/ARZ2020_emhad_noise.yaml/G03generate_events_full_surface_sim/LPDA_2of4_100Hz/4LPDA_1dipole_fullband/noiseless"

#Category separated noise data
datapath = " /mnt/md0/data/SouthPole/single_surface_4LPDA_PA_15m_RNOG_fullsim.json/ARZ2020_emhad_noise.yaml/G03generate_events_full_surface_sim/LPDA_2of4_100Hz/4LPDA_1dipole_fullband/em_had_separately"

#Original "old" data
#datapath = "/mnt/md0/data/SouthPole/single_surface_4LPDA_PA_15m_RNOG_fullsim.json/ARZ2020_emhad_noise.yaml/G03generate_events_full_surface_sim/LPDA_2of4_100Hz/4LPDA_1dipole_fullband" 

n_files = 10
# n_files = 10
n_files_test = 1
norm = 1e-6
n_files_val = int(0.1 * n_files)
# n_files_val = 10
n_files_train = n_files - n_files_val - n_files_test
list_of_file_ids_train = np.arange(n_files_train, dtype=np.int)
list_of_file_ids_val = np.arange(n_files_train, n_files_train + n_files_val, dtype=np.int)
list_of_file_ids_test = np.arange(n_files_train + n_files_val, n_files, dtype=np.int)
n_events_per_file = 100000
batch_size = 64 #Original value was 64

print(f"training on {n_files_train} files ({n_files_train/n_files*100:.1f}%), validating on {n_files_val} files ({n_files_val/n_files*100:.1f}%), testing on {n_files_test} files ({n_files_test/n_files*100:.1f}%)")

steps_per_epoch = n_files_train * (n_events_per_file // batch_size)
n_batches_per_file = n_events_per_file // batch_size
print(f"steps_per_epoch {steps_per_epoch}, n_batches_per_file {n_batches_per_file}")


def spherical_to_cartesian(zenith, azimuth):
    sinZenith = np.sin(zenith)
    x = sinZenith * np.cos(azimuth)
    y = sinZenith * np.sin(azimuth)
    z = np.cos(zenith)
    if hasattr(zenith, '__len__') and hasattr(azimuth, '__len__'):
        return np.array([x, y, z]).T
    else:
        return np.array([x, y, z])


def load_file(i_file, noise=True, norm=norm):


    if noise == True:
       #     t0 = time.time()
       # print(f"loading file {i_file}", flush=True)

	#EM+hadronic event data
        datapath = " /mnt/md0/data/SouthPole/single_surface_4LPDA_PA_15m_RNOG_fullsim.json/ARZ2020_emhad_noise.yaml/G03generate_events_full_surface_sim/LPDA_2of4_100Hz/4LPDA_1dipole_fullband/em_had_separately"

	    #Hadronic event data
		#data = np.load(os.path.join(datapath, f"data_had_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_{i_file:04d}.npy"), allow_pickle=True)[:, :, :, np.newaxis]
        #labels_tmp = np.load(os.path.join(datapath, f"labels_had_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_{i_file:04d}.npy"), allow_pickle=True)
        
		
	#"Old" data set, use together with corresponding datapath
    elif noise == False:
        datapath = "/mnt/md0/data/SouthPole/single_surface_4LPDA_PA_15m_RNOG_fullsim.json/ARZ2020_emhad_noise.yaml/G03generate_events_full_surface_sim/LPDA_2of4_100Hz/4LPDA_1dipole_fullband/noiseless"

    data = np.load(os.path.join(datapath, f"data_emhad_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_{i_file:04d}.npy"), allow_pickle=True)[:, :, :, np.newaxis]
    labels = np.load(os.path.join(datapath, f"labels_emhad_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_{i_file:04d}.npy"), allow_pickle=True)

    return data, labels

#		data = np.load(os.path.join(datapath, f"data_1-3_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_{i_file:04d}.npy"), allow_pickle=True)[:, :, :, np.newaxis]
#        labels_tmp = np.load(os.path.join(datapath, f"labels_1-3_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_{i_file:04d}.npy"), allow_pickle=True)
		
#        shower_energy_em = np.array(labels_tmp.item()["shower_energy_em"])
#       mask = shower_energy_em == 0
#
#	    label_onehot = np.zeros((len(shower_energy_em), 2), dtype=np.int)
#	    label_onehot[mask, 0] = 1
#      	label_onehot[~mask, 1] = 1
#
#    	# check for nans and remove them
#        idx = ~(np.isnan(data))
#        idx = np.all(idx, axis=1)
#        idx = np.all(idx, axis=1)
#        idx = np.all(idx, axis=1)
#	    data = data[idx, :, :, :]
#        data /= norm
   	 #     print(f"finished processing file {i_file} in {time.time() - t0}s")

#       	return data, label_onehot[idx, :]

class TrainDataset(tf.data.Dataset):

    def _generator(file_id):
#         print(f"\nTrain generator current id {file_id}, opening file {list_of_file_ids_train[file_id]}")
        if((file_id + 1) == n_files_train):
#             print("reshuffling")
            np.random.shuffle(list_of_file_ids_train)

        # Opening the file
        i_file = list_of_file_ids_train[file_id]
        data, nu_direction = load_file(i_file, norm)
        num_samples = data.shape[0]
        rand_ids = np.arange(num_samples, dtype=np.int)
        np.random.shuffle(rand_ids)
        n_batches = num_samples // batch_size
        for i_batch in range(n_batches):
            # Reading data (line, record) from the file
            y = nu_direction[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size]]
            x = data[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size], :, :, :]
            yield x, y

    def __new__(cls, file_id):
#         print(f"input arg {tmp}, {batch_size}")
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float64, tf.dtypes.float64),
            output_shapes=((64, 5, 512, 1), (64, 2)), #Original was 64 to match batch size of 64 
            args=(file_id,)
        )


class ValDataset(tf.data.Dataset):

    def _generator(file_id):
#         print(f"\nVal generator current id {file_id}, opening file {list_of_file_ids_val[file_id]}")
        if((file_id + 1) == n_files_val):
            # print("reshuffling")
            np.random.shuffle(list_of_file_ids_val)

        # Opening the file
        i_file = list_of_file_ids_val[file_id]
        data, nu_direction = load_file(i_file, norm)
        num_samples = data.shape[0]
        rand_ids = np.arange(num_samples, dtype=np.int)
        np.random.shuffle(rand_ids)
        n_batches = num_samples // batch_size
        for i_batch in range(n_batches):
            # Reading data (line, record) from the file
            y = nu_direction[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size]]
            x = data[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size], :, :, :]
            yield x, y

    def __new__(cls, file_id):
#         print(f"input arg {tmp}, {batch_size}")
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float64, tf.dtypes.float64),
            output_shapes=((64, 5, 512, 1), (64, 2)), #Original was 64, increased batch size to see effects on training.
            args=(file_id,)
        )

