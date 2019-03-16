
import weightLoader as wl
import generateAdjacencyMatrix as adj
import preprocessing as pp
import filtration

import os

config = {

	"root" : "/Users/kunaalsharma/Desktop/MATH 494/MATH 493/ShapeOfLearning/",

	"symname" : "DigitsCuck",

	"layerWidths" : [420,69],

	"epochs" : [i for i in range(50)],

	"layerNames" : ["cuck1","cuck2"],

	"numProcesses" : 3

}

def bulkProcess():
	'''
	Takes a bunch of arguments

	Runs a pool using one of the two others
		- train a network
		- run the pipeline
	'''
	pass

'''
Trains a network and saves the data.
'''
def trainNetwork():
	#network = ??


'''
Runs the full pipeline for a given epoch. This function can be put into
a pool.
'''
def runPipeline(epoch):
	weights = wl.loadWeights(nnSavePath(epoch),config["layerNames"])
	adjacencyMatrix = adj.getWeightedAdjacencyMatrixNoBias(weights)
	processedMatrix = pp.standardVR(adjacencyMatrix)
	filtration.VR(vrSavePath(epoch),processedMatrix)

	

def runAnalysis():
	'''
	Run cumulative analysis
	'''

'''
Makes a directory for nnData if there isn't one.
Returns the path to the specific epoch
'''
def nnSavePath(epoch):
	path = config["root"] + "data/"+config["symname"] +"/"
	if not os.path.isdir(path):
		os.mkdir(path)

	path = path + str(epoch)
	for weight in config["layerWidths"]:
		path = path + "_" + str(weight)
	return path 

'''
Makes a directory for vrSave if there isn't one.
Returns the path to the specific epoch
'''
def vrSavePath(epoch):
	path = config["root"] + "bettiData/"+config["symname"] + "/" 
	if not os.path.isdir(path):
		os.mkdir(path)

	path = path + str(epoch)
	for weight in config["layerWidths"]:
		path = path + "_" + str(weight)
	return path

if __name__ == "__main__":
	pass
