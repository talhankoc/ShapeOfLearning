import pickle
import numpy as np

path = '/Users/tkoc/Code/ShapeOfLearning/Homology/BettiDataFloydVR/'
file_name = 'VRFiltration_BettiData.txt'
folder_prefix = 'Fashion2_'
layer_sizes = [8,16,24,32,40,48]
layer_xy_values = []
epochs = 50

for layer_size in layer_sizes:
	max_x = 0
	min_x = float('inf')
	for epoch in range(1, epochs+1):
		folder_path = path + folder_prefix + str(epoch) + '_' + str(layer_size) + '/'
		data_path = folder_path + file_name
		with open(data_path, 'rb') as f:
			diagrams = pickle.load(f)
			diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]
			concat_dgms = np.concatenate(diagrams).flatten()
			finite_dgms = concat_dgms[np.isfinite(concat_dgms)]
			ax_max, ax_min = np.max(finite_dgms), np.min(finite_dgms)
			if ax_max > max_x:
				max_x = ax_max
			if ax_min < min_x:
				min_x = ax_min
	layer_xy_values.append((max_x,min_x))

print(layer_xy_values)
