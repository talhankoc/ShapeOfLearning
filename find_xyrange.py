import pickle
import numpy as np

def run(load_paths):
	max_x = 0
	min_x = float('inf')
	for fn in load_paths:
		with open(fn, 'rb') as f:
			diagrams = pickle.load(f)
			diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]
			betti1 = diagrams[1]
			betti1 = np.concatenate(betti1).flatten()
			finite_dgms = betti1[np.isfinite(betti1)]
			ax_max, ax_min = np.max(finite_dgms), np.min(finite_dgms)
			print(f'{fn[-2:]}', ax_max, ax_min)
			max_x = max(max_x, ax_max)
			min_x = min(min_x, ax_min)

	print(f'max={max_x}')
	return max_x
