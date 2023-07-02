import os
import numpy as np
import csv

current_path = os.path.dirname(__file__)

path = f"{current_path}/dataset/lungcancer.csv"

data = np.loadtxt(path, dtype=np.float32, delimiter=",")

print(data[:53, 16].shape)