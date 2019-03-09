import h5py

# input: fn, relative to directory
#		 layer_names, ex: ['dense_1']
# output: list of np arrays that contain the weights from each layer
def loadWeights(fn, layer_names):
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

#print(loadWeights('NNGeneration/Saved Models/CIFAR-10-AML/model-01.hdf5', ['dense_1']))


# helper function to take a look at what the layer names are
def geth5py(fn):
	print(f'loading weights from saved model {fn} \n ...')
	W = []
	f = h5py.File(fn, 'r')
	model_weights = f['model_weights']
	return model_weights


