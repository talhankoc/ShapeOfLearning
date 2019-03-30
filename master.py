import weightLoader as wl
import generateAdjacencyMatrix as adj
import preprocessing as pp
import filtration
import os
from multiprocessing import Pool 

'''
Runs all betti generation in parallel using the runPipeline function.
'''
def bulkProcess():
	for epoch in config["epochs"]:
		runPipeline(epoch)
	
'''
Trains any network. The network should take the nnSavePath function as an 
argument, and use that to generate the save location for each epoch. Nothing
else needs to change.
'''
def trainNetwork():
	import NNGeneration.NNDigits as digits
	digits.randomModel(config)


'''
Runs the full pipeline for a given epoch. This function can be put into
a pool.

(Only runs the pipeline for a single epoch intentionally, so that it can 
be run in parallel by a pool.)
'''
def runPipeline(epoch):
	weights = wl.simpleLoader(nnSavePath(epoch))
	adjacencyMatrix = adj.getWeightedAdjacencyMatrixNoBias(weights)
	processedMatrix = pp.standardVR(adjacencyMatrix)
	filtration.VR(vrSavePath(epoch),processedMatrix)

	print(f"Finished epoch: {epoch}")

	
'''
Analysis is anything that runs using all betti information (for example, 
gradient.py or delta.py). Any analysis should have a function "runAnalysis"
that accepts a list of paths to vrData and the analysisSavePath.
'''
def runAnalysis():
	import analysis
	'''
	vrPaths = []
	imageSavePaths = []
	for epoch in config["epochs"]:
		vrPaths.append(vrSavePath(epoch, mkdir=False))
		imageSavePaths.append(imageSavePath(epoch))
	
	analysis.runAnalysisAndVisualization(vrPaths,imageSavePaths,analysisBettiDistributionSavePath())
	'''
	analysis.runFloydReplacement(config)

'''
Makes a directory for nnData if there isn't one and mkdir is
specified.
Returns the path to the specific epoch
'''
def nnSavePath(epoch, fstring = False, hdf5 = False, mkdir=True):
	path = config["root"] + "data/" + config["symname"] + "/" + config['nnSaveFnPre']
	if not os.path.isdir(path) and mkdir:
		os.mkdir(path)

	if fstring:
		path = path + "{epoch:d}"
	else:
		path = path + str(epoch)

	for weight in config["layerWidths"]:
		path = path + "_" + str(weight)
	
	if hdf5: 
		return path + ".hdf5"
	return path

'''
Makes a directory for vrSave if there isn't one and mkdir is
specified.
Returns the path to the specific epoch
'''
def vrSavePath(epoch, mkdir=True):
	path = config["root"] + "bettiData/"+config["symname"] + "/" 
	if not os.path.isdir(path) and mkdir:
		os.mkdir(path)

	path = path + str(epoch)
	for weight in config["layerWidths"]:
		path = path + "_" + str(weight)
	return path

'''
Makes a directory for analysisSave if there isn't one and mkdir is 
specified.
Returns the path to the specified symName path.
'''
def analysisSavePath(mkdir=True):
	path = config["root"] + "analysis/" + config["symname"] + '/'
	if not os.path.isdir(path) and mkdir:
		os.mkdir(path)
	return path 

def analysisBettiDistributionSavePath(mkdir=True):
	path = analysisSavePath(mkdir) + 'bettiDistribution/'
	if not os.path.isdir(path) and mkdir:
		os.mkdir(path)
	return path
'''
Makes a directory for testTrainSave if there isn't one and mkdir is 
specified.
Returns the path to the specified symName path.
'''
def accSavePath(mkdir=True):
	path =config["root"] + "data/" + config["symname"] + "/"
	if not os.path.isdir(path) and mkdir:
		os.mkdir(path)
	path = path + "scores"
	for weight in config["layerWidths"]:
		path = path + "_" + str(weight)
	return path + ".txt"

def imageSavePath(epoch):
	return analysisBettiDistributionSavePath() + f'Epoch{epoch}_birth_death.png'

def plotModelMetrics():
	import scorePlot
	scorePlot.run(accSavePath(mkdir=False),config['epochs'],analysisSavePath(), close_up_range=0)
	scorePlot.run(accSavePath(mkdir=False),config['epochs'],analysisSavePath(), close_up_range=4)

def makeGIF():
	import gifMake
	loadPaths = []
	for epoch in config["epochs"]:
		loadPaths.append(imageSavePath(epoch))
	gifMake.run(loadPaths, analysisSavePath() + 'birthDeath.gif')

config = {

	"root" : "/Users/kunaalsharma/Desktop/MATH 494/MATH 493/ShapeOfLearning/",

	"symname" : "DigitsSimple",

	"layerWidths" : [8],

	"epochs" : [i for i in range(1,51)],

	"layerNames" : ["Dense",],
	
	"numProcesses" : 1,

	"nnSaveFn" : nnSavePath,

	"accSaveFn" : accSavePath,

	"analysisSaveFn": analysisSavePath,

	"inputSize" : 784,

	"outputSize" : 10,

	"nnSaveFnPre":""#'MODEL_Epoch'
}
'''
Change this depending upon what you want to do
'''
if __name__ == "__main__":
	runAnalysis()



