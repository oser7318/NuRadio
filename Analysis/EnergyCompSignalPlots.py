import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
from generator import load_file, list_of_file_ids_test, batch_size



#---------------------------------------------------------------------
i_file = list_of_file_ids_test[0]

print("Done!\n Loading labels...")

datapath = "/mnt/ssd2/data/SouthPole/single_surface_4LPDA_PA_15m_RNOG_fullsim.json/ARZ2020_emhad_noise.yaml/G03generate_events_full_surface_sim/LPDA_2of4_100Hz/4LPDA_1dipole_fullband/em_had_separately"
labels_em = np.load(os.path.join(datapath, f"labels_emhad_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_{i_file:04d}.npy"), allow_pickle=True)
labels_had = np.load(os.path.join(datapath, f"labels_had_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_{i_file:04d}.npy"), allow_pickle=True)

print("Done!\n Loading data files...")

#TRY NOISELESS DATA AND SEE IF HIGH AND LOW EM SHOWER ENERGY EVENTS DIFFER. 
#MAYBE LOW EM SHOWER ENERGY EVENTS LOOK MORE LIKE HADRONIC SHOWERS. 

data, category = load_file(i_file, noise=False, em=True)


#Plot singlas: High vs low EM shower energy, high vs low total shower energy. 

#EM component energies for the e CC events
em_shower_energy = np.array(labels_em.item()["shower_energy_em"])
low_energy_indeces = np.where(em_shower_energy < 1e16)
low_energy_plot_indeces = low_energy_indeces[0][0] #250 12

high_energy_indeces = np.where(em_shower_energy > 10**18.5)
high_energy_plot_indeces = high_energy_indeces[0][0]

xaxis = np.linspace(0, 256, 512)

fig, ax = plt.subplots(5, 2, sharex=True)

for i in range(5):
    ax[i, 0].plot(xaxis, data[low_energy_plot_indeces, i])
    ax[i, 1].plot(xaxis, data[high_energy_plot_indeces, i])
    #ax[i, 2].plot(xaxis, data[low_energy_plot_indeces[2], i])
    
    # ax[i, :].set_ylabel('V')
    # ax[i, :].set_xlabel('ns')

    if i < 4:
        ax[i, 0].set_title(f'LPDA{i+1}')
        ax[i, 1].set_title(f'LPDA{i+1}')
        #ax[i, 2].set_title(f'LPDA{i+1}')
    elif i == 4:
        ax[i, 0].set_title('Dipole')
        ax[i, 1].set_title('Dipole')
        #ax[i, 2].set_title('Dipole')

fig.set_size_inches(12,7)
fig.tight_layout()
plt.savefig('HighvsLow_E_Events_NoNoise.png', dpi=200)



