import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

#Runs
runs = ("(1,3)", "(1,5)", "(1,10)", "(1,15)", "(1,20)")
acc = [73.58, 74.46, 71.95, 75.60, 68.44]

bars = ax.barh(runs, acc, align="center")
ax.set_yticks(np.arange(len(runs)))
ax.set_yticklabels(runs)
ax.set_xlim(50)
ax.bar_label(bars)

ax.set_xlabel("Best val_accuracy (%)")
ax.set_title("Accuracy dependence on kernel size")

plt.show()