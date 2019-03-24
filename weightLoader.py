import numpy as np
import h5py
'''
Arguments:
	path : base path to the file 

Output:
	W : list of weights 
'''
def customLoader(path, layer_numbers):
	print('Loading weights with custom method.')
	W = []
	for l in layer_numbers:
		print(f'\t{path}_W{l}.npy')
		W.append(np.load(f'{path}_W{l}.npy'))
	return W
	
'''
Arguments: 
	fn : path to h5py file
	layer_names : list of layer names (strings)

Output:
	W : list of weights
'''
def loadWeights(fn, layer_names):
	fn = fn + ".hd5"
	print(f'loading weights from saved model {fn} \n ...')
	W = []
	f = h5py.File(fn, 'r')
	#print(list(f.keys()))
	model_weights = f['model_weights']
	#print(list(model_weights.keys()))
	for X in layer_names:
		w = model_weights[X][X]['kernel:0'].value
		W.append(w)
	f.close()
	return W

'''
Loads weights saved by keras_model_saver. The provided path 
is for a single epoch. Stupid implementation but oh well *shrugs*
'''
def simpleLoader(path):
	W = []
	for i in range(1,1000):
		try:
			currData = np.load(path + f"_W{i}.npy")
			W.append(currData)
		except:
			break
	return W



