
import weightLoader as wl
import generateAdjacencyMatrix as adj
import preprocessing as pp
import filtration

import os

from multiprocessing import Pool 

config = {

	"root" : "/Users/kunaalsharma/Desktop/MATH 494/MATH 493/ShapeOfLearning/",

	"symname" : "DigitsCuck",

	"layerWidths" : [420,69],

	"epochs" : [i for i in range(50)],

	"layerNames" : ["cuck1","cuck2"],

	"numProcesses" : 3

}

'''
Runs all betti generation in parallel using the runPipeline function.
'''
def bulkProcess():
	pool = Pool(processes = config["numProcesses"])
	pool.map(runPipeline,config["epochs"])

'''
Trains any network. The network should take the nnSavePath function as an 
argument, and use that to generate the save location for each epoch. Nothing
else needs to change.
'''
def trainNetwork():
	import sampleNetwork
	sampleNetwork.makeAndRun(nnSavePath)


'''
Runs the full pipeline for a given epoch. This function can be put into
a pool.

(Only runs the pipeline for a single epoch intentionally, so that it can 
be run in parallel by a pool.)
'''
def runPipeline(epoch):
	weights = wl.loadWeights(nnSavePath(epoch),config["layerNames"])
	adjacencyMatrix = adj.getWeightedAdjacencyMatrixNoBias(weights)
	processedMatrix = pp.standardVR(adjacencyMatrix)
	filtration.VR(vrSavePath(epoch),processedMatrix)

	
'''
Analysis is anything that runs using all betti information (for example, 
gradient.py or delta.py). Any analysis should have a function "runAnalysis"
that accepts a list of paths to vrData and the analysisSavePath.
'''
def runAnalysis():
	paths = []
	for epoch in config["epochs"]:
		paths.append(vrSavePath(epoch))

	import sampleAnalysis
	sampleAnalysis.runAnalysis(paths,analysisSavePath())

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

'''
Makes a directory for analysisSave if there isn't one.
Returns the path to the specified symName path.
'''
def analysisSavePath():
	path = config["root"] + "analysis/" + config["symname"] + "/"
	if not os.path.isdir(path):
		os.mkdir(path)
	return path 

'''
Change this depending upon what you want to do
'''
if __name__ == "__main__":
	bulkProcess()