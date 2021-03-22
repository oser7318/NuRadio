import numpy as np 
import matplotlib.pyplot as plt
import itertools
from generator import load_file
import os
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True) 

datapath = "/mnt/md0/data/SouthPole/single_surface_4LPDA_PA_15m_RNOG_fullsim.json/ARZ2020_emhad_noise.yaml/G03generate_events_full_surface_sim/LPDA_2of4_100Hz/4LPDA_1dipole_fullband"

data, labels = load_file(10)

One, TheOther = np.bincount(labels)

total = Hadr + EMplusHadr
print('Examples:\n    Total: {}\n    One: {} ({:.2f}% of total)\n'.format(
    total, Hadr, 100 * pos / total))