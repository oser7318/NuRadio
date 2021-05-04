import os
import numpy as np
import tensorflow as tf
import time
from sklearn.utils import shuffle

np.set_printoptions(precision=4)

#Category separated Noiseless data
#datapath = "/mnt/md0/data/SouthPole/single_surface_4LPDA_PA_15m_RNOG_fullsim.json/ARZ2020_emhad_noise.yaml/G03generate_events_full_surface_sim/LPDA_2of4_100Hz/4LPDA_1dipole_fullband/noiseless"

#Category separated noise data
#datapath = " /mnt/md0/data/SouthPole/single_surface_4LPDA_PA_15m_RNOG_fullsim.json/ARZ2020_emhad_noise.yaml/G03generate_events_full_surface_sim/LPDA_2of4_100Hz/4LPDA_1dipole_fullband/em_had_separately"

#Original "old" data
#datapath = "/mnt/md0/data/SouthPole/single_surface_4LPDA_PA_15m_RNOG_fullsim.json/ARZ2020_emhad_noise.yaml/G03generate_events_full_surface_sim/LPDA_2of4_100Hz/4LPDA_1dipole_fullband" 

n_files = 7
# n_files = 10
n_files_test = 1
norm = 1e-6
n_files_train = 5 #int(0.8 * n_files)
# n_files_val = 10
n_files_val = 1 #n_files - n_files_train - n_files_test
list_of_file_ids_train = np.arange(n_files_train, dtype=np.int)
list_of_file_ids_val = np.arange(n_files_train, n_files_train + n_files_val, dtype=np.int)
list_of_file_ids_test = np.arange(n_files_train + n_files_val, n_files, dtype=np.int)
n_events_per_file = 200000
batch_size = 64 #Original value was 64

Noise=True

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


def load_file(i_file, noise=True, em=True, norm=norm):


    if noise:
       #     t0 = time.time()
       # print(f"loading file {i_file}", flush=True)

	#EM+hadronic event data
        datapath = "/mnt/ssd2/data/SouthPole/single_surface_4LPDA_PA_15m_RNOG_fullsim.json/ARZ2020_emhad_noise.yaml/G03generate_events_full_surface_sim/LPDA_2of4_100Hz/4LPDA_1dipole_fullband/em_had_separately"

	    #Hadronic event data
		#data = np.load(os.path.join(datapath, f"data_had_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_{i_file:04d}.npy"), allow_pickle=True)[:, :, :, np.newaxis]
        #labels_tmp = np.load(os.path.join(datapath, f"labels_had_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_{i_file:04d}.npy"), allow_pickle=True)


    elif not noise:
        datapath = "/mnt/ssd2/data/SouthPole/single_surface_4LPDA_PA_15m_RNOG_fullsim.json/ARZ2020_emhad_noise.yaml/G03generate_events_full_surface_sim/LPDA_2of4_100Hz/4LPDA_1dipole_fullband/noiseless"

    if em:

        data = np.load(os.path.join(datapath, f"data_emhad_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_{(i_file):04d}.npy"), allow_pickle=True)[:, :, :, np.newaxis]
        labels_tmp = np.load(os.path.join(datapath, f"labels_emhad_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_{(i_file):04d}.npy"), allow_pickle=True)

    elif not em:

        data = np.load(os.path.join(datapath, f"data_had_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_{(i_file):04d}.npy"), allow_pickle=True)[:, :, :, np.newaxis]
        labels_tmp = np.load(os.path.join(datapath, f"labels_had_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_{(i_file):04d}.npy"), allow_pickle=True) #set 3+i_file to check if file 0001 is broken
        
	#"Old" data set, use together with corresponding datapath
#		data = np.load(os.path.join(datapath, f"data_1-3_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_{i_file:04d}.npy"), allow_pickle=True)[:, :, :, np.newaxis]
#       labels_tmp = np.load(os.path.join(datapath, f"labels_1-3_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_{i_file:04d}.npy"), allow_pickle=True)
    
    shower_energy_em = np.array(labels_tmp.item()["shower_energy_em"])
    mask = shower_energy_em == 0

    label_onehot = np.zeros((len(shower_energy_em), 2), dtype=np.int)
    label_onehot[mask, 0] = 1
    label_onehot[~mask, 1] = 1

    	# check for nans and remove them
    idx = ~(np.isnan(data))
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    idx = np.all(idx, axis=1)
    data = data[idx, :, :, :]
    data /= norm
    
    print(f'Data shape: {data.shape}\t Labels shape: {label_onehot[idx, :].shape}')
    
    return data, label_onehot[idx, :]

# #Shuffle function that alters states of a and b in the same way, no copies created.
# def shuffle_same(a,b):
    
#     rng_state = np.random.get_state()
#     np.random.shuffle(a)
#     np.random.set_state(rng_state)
#     np.random.shuffle(b)

def TestDataset(noise=False):
    
        i_file = list_of_file_ids_test[0]

        if noise: 
            data_had, labels_had = load_file(i_file, noise=True, em=False, norm=norm) #Choose noisy or noiseless data
            data_emhad, labels_emhad = load_file(i_file, noise=True, em=True, norm=norm) 

        elif not noise:
            data_had, labels_had = load_file(i_file, noise=False, em=False, norm=norm) #Choose noisy or noiseless data
            data_emhad, labels_emhad = load_file(i_file, noise=False, em=True, norm=norm) 

        #Joint data array (not shuffeled and doubble size of constituent data)
        data_combined = np.concatenate( (data_had, data_emhad), axis=0)
        labels_combined = np.concatenate( (labels_had, labels_emhad), axis=0)

        #Shuffle using shuffle_same(a,b)
        #shuffle_same(data_combined,labels_combined)

        #data_combined = data_combined[0:10000]
        #labels_combined = labels_combined[0:10000]

         #Shuffle using sklearn.utils.shuffle, which leaves the imput array intact (creates a copy but shuffeled)
        #data_combined, labels_combined = shuffle(data_combined, labels_combined, random_state=0)

        return data_combined, labels_combined

class TrainDatasetEven(tf.data.Dataset):

    def _generator(file_id):
        if((file_id + 1) == n_files_train):
#             print("reshuffling")
            np.random.shuffle(list_of_file_ids_train)

        i_file = list_of_file_ids_train[file_id]
        data_had, labels_had = load_file(i_file, noise=Noise, em=False, norm=norm) #Choose noisy or noiseless data
        data_emhad, labels_emhad = load_file(i_file, noise=Noise, em=True, norm=norm) 

        #Joint data array (not shuffeled and doubble size of constituent data)
        data_combined = np.concatenate( (data_had, data_emhad), axis=0)
        labels_combined = np.concatenate( (labels_had, labels_emhad), axis=0)

        #Shuffle using shuffle_same(a,b)
        #shuffle_same(data_combined,labels_combined)

        #Shuffle using sklearn.utils.shuffle, which leaves the imput array intact (creates a copy but shuffeled)
        #data_combined, labels_combined = shuffle(data_combined, labels_combined, random_state=0)

        #data_combined = data_combined[0:100000]
        #labels_combined = labels_combined[0:100000]

        num_samples = data_combined.shape[0]
        rand_ids = np.arange(num_samples, dtype=np.int)
        np.random.shuffle(rand_ids)
        n_batches = num_samples // batch_size
        for i_batch in range(n_batches):
            # Reading data (line, record) from the file
            y = labels_combined[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size]]
            x = data_combined[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size], :, :, :]
            yield x, y

    def __new__(cls, file_id):
#         print(f"input arg {tmp}, {batch_size}")
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float64, tf.dtypes.float64),
            output_shapes=((batch_size, 5, 512, 1), (batch_size, 2)), #Original was 64 to match batch size of 64 
            args=(file_id,)
        )
    

class ValDatasetEven(tf.data.Dataset):

    def _generator(file_id):
        if((file_id + 1) == n_files_val):
#             print("reshuffling")
            np.random.shuffle(list_of_file_ids_val)

        i_file = list_of_file_ids_val[file_id]
        data_had, labels_had = load_file(i_file, noise=Noise, em=False, norm=norm) #Choose noisy or noiseless data
        data_emhad, labels_emhad = load_file(i_file, noise=Noise, em=True, norm=norm) 

        #Joint data array (not shuffeled and doubble size of constituent data)
        data_combined = np.concatenate( (data_had, data_emhad), axis=0)
        labels_combined = np.concatenate( (labels_had, labels_emhad), axis=0)

        #Shuffle using shuffle_same(a,b)
        #shuffle_same(data_combined,labels_combined)

        #Shuffle using sklearn.utils.shuffle, which leaves the imput array intact (creates a copy but shuffeled)
        #data_combined, labels_combined = shuffle(data_combined, labels_combined, random_state=0)

       # data_combined = data_combined[0:100000]
       # labels_combined = labels_combined[0:100000]

        num_samples = data_combined.shape[0]
        rand_ids = np.arange(num_samples, dtype=np.int)
        np.random.shuffle(rand_ids)
        n_batches = num_samples // batch_size
        for i_batch in range(n_batches):
            # Reading data (line, record) from the file
            y = labels_combined[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size]]
            x = data_combined[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size], :, :, :]
            yield x, y

    def __new__(cls, file_id):
#         print(f"input arg {tmp}, {batch_size}")
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float64, tf.dtypes.float64),
            output_shapes=((batch_size, 5, 512, 1), (batch_size, 2)), #Original was 64 to match batch size of 64 
            args=(file_id,)
        )

#ctrl+k+u to uncomment

# class TrainDataset(tf.data.Dataset):

#     def _generator(file_id):
# #         print(f"\nTrain generator current id {file_id}, opening file {list_of_file_ids_train[file_id]}")
#         if((file_id + 1) == n_files_train):
# #             print("reshuffling")
#             np.random.shuffle(list_of_file_ids_train)

#         # Opening the file
#         i_file = list_of_file_ids_train[file_id]
#         data, nu_direction = load_file(i_file, norm)
#         num_samples = data.shape[0]
#         rand_ids = np.arange(num_samples, dtype=np.int)
#         np.random.shuffle(rand_ids)
#         n_batches = num_samples // batch_size
#         for i_batch in range(n_batches):
#             # Reading data (line, record) from the file
#             y = nu_direction[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size]]
#             x = data[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size], :, :, :]
#             yield x, y

#     def __new__(cls, file_id):
# #         print(f"input arg {tmp}, {batch_size}")
#         return tf.data.Dataset.from_generator(
#             cls._generator,
#             output_types=(tf.dtypes.float64, tf.dtypes.float64),
#             output_shapes=((64, 5, 512, 1), (64, 2)), #Original was 64 to match batch size of 64 
#             args=(file_id,)
#         )


# class ValDataset(tf.data.Dataset):

#     def _generator(file_id):
# #         print(f"\nVal generator current id {file_id}, opening file {list_of_file_ids_val[file_id]}")
#         if((file_id + 1) == n_files_val):
#             # print("reshuffling")
#             np.random.shuffle(list_of_file_ids_val)

#         # Opening the file
#         i_file = list_of_file_ids_val[file_id]
#         data, nu_direction = load_file(i_file, norm)
#         num_samples = data.shape[0]
#         rand_ids = np.arange(num_samples, dtype=np.int)
#         np.random.shuffle(rand_ids)
#         n_batches = num_samples // batch_size
#         for i_batch in range(n_batches):
#             # Reading data (line, record) from the file
#             y = nu_direction[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size]]
#             x = data[rand_ids[i_batch * batch_size:(i_batch + 1) * batch_size], :, :, :]
#             yield x, y

#     def __new__(cls, file_id):
# #         print(f"input arg {tmp}, {batch_size}")
#         return tf.data.Dataset.from_generator(
#             cls._generator,
#             output_types=(tf.dtypes.float64, tf.dtypes.float64),
#             output_shapes=((64, 5, 512, 1), (64, 2)), #Original was 64, increased batch size to see effects on training.
#             args=(file_id,)
#         )

