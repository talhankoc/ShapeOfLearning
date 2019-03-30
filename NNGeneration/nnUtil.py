import numpy as np 
import tensorflow as tf 
import keras 

'''
Calls randomUpdate on each layer in the model
'''
def randomUpdateModel(model):
	for layer in model.layers[1:]:
		randomUpdate(layer)
	return model

'''
Replaces each param in a layer with a float drawn 
from the standard normal distribution
'''
def randomUpdate(layer):
	weights,bias = layer.get_weights()
	(rows,cols) = np.shape(weights)
	for i in range(rows):
		for j in range(cols):
			weights[i,j] = np.random.normal()
	for i in range(len(bias)):
		bias[i] = np.random.normal()
	layer.set_weights(np.array([weights,bias]))

'''
Inserts weights and biases into a model. Returns the model.
'''
def constructModel(model,weights,biases):
	combinedParams = zip(weights,biases)
	for newParams,layer in zip(combinedParams,model.layers[1:]):
		layer.set_weights(np.array(newParams))
	return model

'''
Loads all the biases for a give model, given the bias path.
'''
def loadBias(biasPath, numBias):
	biases = []
	for i in range(1,numBias+1):
		currBias = np.load(biasPath + f"_b{i}.npy")
		biases.append(currBias)
	return biases

'''
Returns the accuracy of a given model on a given dataset.
'''
def getAccuracy(model,x,y):
	return model.evaluate(x,y,verbose=0)[1]


'''
Takes a list of layer sizes and a matrix, and returns a list of 
weight matrices.
'''
def extractWeightsFromMatrix(layerSizes,matrix):
	rowOffset = 0
	colOffset = layerSizes[0]
	weights = []

	for i in range(len(layerSizes)-1):
		currWeights = extract(matrix,rowOffset,layerSizes[i],colOffset,layerSizes[i+1])
		weights.append(currWeights)
		rowOffset += layerSizes[i]
		colOffset += layerSizes[i+1]

	return weights
	
'''
Returns the submatrix of a symmetrical adjacency matrix corresponding to 
the weights of a layer. Eg: with 784 input neurons and a hidden layer of size 10
you want rowOffset = 0, rowSize = 784, colOffset = 784, colSize = 10
'''
def extract(matrix,rowOffset,rowSize,colOffset,colSize):
	return matrix[rowOffset:rowOffset + rowSize,colOffset:colOffset + colSize]

'''
Replace the weights and biases in a model with those created by Floyd Warshall.
Loads biases from file, and removes weights from adjacency matrix. Returns new
model.
'''
def replaceModelParams(model,epoch,matrix,config):
	weightList = config["layerWidths"].copy()
	weightList.insert(0,config["inputSize"])
	weightList.append(config["outputSize"])
	numBias = len(weightList) - 1 
	weights = extractWeightsFromMatrix(weightList,matrix)
	biases = loadBias(config["nnSaveFn"](epoch),numBias)
	return constructModel(model,weights,biases)


'''
Returns mnist training and testing data
'''
def mnistTrainTest():
	mnist = tf.keras.datasets.mnist
	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0
	return (x_train, y_train, x_test, y_test)











