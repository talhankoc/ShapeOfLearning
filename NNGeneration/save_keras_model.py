import numpy as np

def save_model(model, fn):
	i = 0
	for layer in model.layers:
		weights = layer.get_weights() # list of numpy arrays
		if len(weights) == 2:
			i += 1
			W = np.array(weights[0])
			b = np.array(weights[1])
			np.save(f"{fn}_W"+str(i),W)
			np.save(f"{fn}_b"+str(i),b)