#Plot simulated detector data, both with and without noise.

import numpy as np 
import matplotlib.pyplot as plt 
import itertools
import os
from generator import load_file

data, labels = load_file(10)

print(ndarray.shape(data))