import pickle
import numpy as np

path = 'Homology/Data/CIFAR-10-Variation2/'
#file_name = 'VRFiltration_BettiData.txt'
fn1 = 'VRFiltration_BettiData.txt'
folder_prefix = 'model-'
epochs = range(101)

max_x = 0
min_x = float('inf')
for epoch in epochs:
	folder_path = path + folder_prefix + str(epoch) + '/'
	fn = folder_path + fn1
	with open(fn, 'rb') as f:
		diagrams = pickle.load(f)
		diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]
		concat_dgms = np.concatenate(diagrams).flatten()
		finite_dgms = concat_dgms[np.isfinite(concat_dgms)]
		ax_max, ax_min = np.max(finite_dgms), np.min(finite_dgms)
		# x_r = ax_max - ax_min

		# # Give plot a nice buffer on all sides.
		# # ax_range=0 when only one point,
		# buffer = 1 if xy_range == 0 else x_r / 5

		# x_down = ax_min - buffer / 2
		# x_up = ax_max + buffer
		# y_down, y_up = x_down, x_up

		print(f'Epoch {epoch}', ax_max, ax_min)
		max_x = max(max_x, ax_max)
		min_x = min(min_x, ax_min)

print(max_x, min_x)
