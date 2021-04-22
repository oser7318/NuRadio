import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

#Runs
runs = ("Noiseless", "Noise")
acc = [95.65, 76.48]

#Fluctuation in acc for 5 sessions of 4.2.0 training on 20 files. (1,7) kernel size
best_acc = [75.90, 76.08, 75.94, 75.25, 76.25]

bars = ax.barh(runs, acc, align="center")
ax.set_yticks(np.arange(len(runs)))
ax.set_yticklabels(runs)
ax.set_xlim(50)
ax.bar_label(bars)

ax.set_xlabel("Test accuracy (%)")
ax.set_title("Accuracy of NN on  noisy and noiseless data")

plt.show()