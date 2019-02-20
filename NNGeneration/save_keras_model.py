import numpy as np

def save_model(model, location, name):
	i = 0
	for layer in model.layers:
		weights = layer.get_weights() # list of numpy arrays
		if len(weights) == 2:
			i += 1
			W = np.array(weights[0])
			b = np.array(weights[1])
			np.save(location+name+"_W"+str(i),W)
			np.save(location+name+"_b"+str(i),b)