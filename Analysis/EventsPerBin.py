import matplotlib.pyplot as plt
import numpy as np
import os
from math import log10
datapath = "/mnt/ssd2/data/SouthPole/single_surface_4LPDA_PA_15m_RNOG_fullsim.json/ARZ2020_emhad_noise.yaml/G03generate_events_full_surface_sim/LPDA_2of4_100Hz/4LPDA_1dipole_fullband/em_had_separately"




nu_energies = np.logspace(17, 19, 17)
counts = np.zeros(len(nu_energies)-1)
cases_em_larger = np.zeros(len(nu_energies)-1)

for i in range(10):

    print(f'Loading file {i+1}')
    labels = np.load(os.path.join(datapath, f"labels_emhad_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_{(i+1):04d}.npy"), allow_pickle=True)
    shower_energy_em = np.array(labels.item()["shower_energy_em"])
    shower_energy_had = np.array(labels.item()["shower_energy_had"])
    nu_energy = np.array(labels.item()["nu_energy"])
    print('Done!')

    for e in range(len(nu_energies)-1):
        indeces = np.where((nu_energy > nu_energies[e]) & (nu_energy < nu_energies[e+1]))
        cases_em_larger[e] += np.sum(shower_energy_em[indeces] > shower_energy_had[indeces])
        counts[e] += len(indeces[0])
        

log_energies = [log10(energy) for energy in nu_energies]
x = [ (log_energies[j]+log_energies[j+1])/2 for j in range(len(log_energies)-1) ]

frac_em_larger = cases_em_larger/counts



#------------PLOT---------------
plt.plot(x, frac_em_larger, 'bo')
plt.xlim(15)
plt.xlabel('log($E_{\u03BD}$)')
plt.ylabel('Fraction')
plt.title(r'Fraction of events where $E_{EM} > E_{HAD}$')

plt.tight_layout()
plt.savefig('FracHadLarger.png')
