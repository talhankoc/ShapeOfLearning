import find_xyrange
import plotBettiCurve 
import pickle
from numpy import inf
import numpy as np
from BettiPlotter import plot_dgms
import os



def runAnalysisAndVisualization(load_paths, imageSavePaths, root_save_path):

	#find range to keep diagrams same scale
	max_x = find_xyrange.run(load_paths)
	bettiAnalysisPaths = {'betti0':[], 'betti1':[]}
	#generate betti diagram for each load_path
	for i, val in enumerate(zip(load_paths,imageSavePaths)):
		load_path, save_path = val
		with open(load_path, 'rb') as f:
			diagrams = pickle.load(f)
		#generateBettiDiagram(diagrams, save_path, i+1, max_x)
		fn0 = f'{root_save_path}Epoch{i+1}_betti0DistributionAnalysis'
		fn1 = f'{root_save_path}Epoch{i+1}_betti1DistributionAnalysis'
		print(f'Saving {fn1}')
		bettiDistributionAnalysis(diagrams[0], fn0)
		bettiDistributionAnalysis(diagrams[1], fn1)
		bettiAnalysisPaths['betti0'].append(fn0)
		bettiAnalysisPaths['betti1'].append(fn1)
	plotBettiCurve.run(bettiAnalysisPaths['betti0'], f'{root_save_path}Betti0Curves/')
	plotBettiCurve.run(bettiAnalysisPaths['betti1'], f'{root_save_path}Betti1Curves/')


def generateBettiDiagram(diagrams, saveFn, i, x_range=None, plot_lifetime_too=False, title=None):
	
	
	plot_dgms(diagrams, 
			size=12,
			title=f'{title} | Epoch {i}',
			save_path=saveFn,
			xy_range=[-1,x_range,-1,x_range] if x_range else None
			)
	
	if plot_lifetime_too:
		plot_dgms(diagrams, 
			size=12, 
			title=f'{title}  | Epoch {i}',
			save_path=saveFn, 
			xy_range=[-1,x_range,-1,x_range] if x_range else None, 
			lifetime=True
			)

def bettiDistributionAnalysis(datapoints, saveFn):
	lifetime = np.array([ [x,y-x] for x,y in datapoints if y != inf ])
	mean, std = np.mean(lifetime[:,1]), np.std(lifetime[:,1])
	with open(saveFn, 'wb') as f:
		pickle.dump(\
			{'mean':mean, 'std':std, 'count':len(lifetime)},\
			f)
'''
Replaces all weights in a model with those generated by FloydWarshall at each epoch.
Find the testing/training accuracy of the model with the new weights, and 
saves them.
'''
def runFloydReplacement(config):
	import NNGeneration.nnUtil as utils
	import NNGeneration.NNDigits as digits
	import weightLoader as wl
	import generateAdjacencyMatrix as adj
	import preprocessing as pp
	from NNGeneration.DigitsPositive import base_model, get_dataset

	#model = digits.emptyModel(config)
	#xTrain,yTrain,xTest,yTest = utils.mnistTrainTest()
	model = base_model()
	xTrain, xTest, yTrain, yTest = get_dataset()
	trainAccs = []
	testAccs = []

	for epoch in config["epochs"]:
		weights = wl.simpleLoader(config["nnSaveFn"](epoch))
		adjacencyMatrix = adj.getWeightedAdjacencyMatrixNoBias(weights.copy())
		processedMatrix = pp.standardVR(adjacencyMatrix)

		model = utils.replaceModelParams(model,epoch,processedMatrix,config, weights)
		train = utils.getAccuracy(model,xTrain,yTrain)
		test = utils.getAccuracy(model,xTest,yTest)

		trainAccs.append(train)
		testAccs.append(test)

		print(f"Finished epoch: {epoch}. Train: {train}, Test: {test}")

	path = config["analysisSaveFn"] + "replacement/"
	if not os.path.isdir(path):
		os.mkdir(path)

	with open(path+"acc.txt","w+") as f:
		for i in range(len(trainAccs)):
			f.write(f"{i},\t{trainAccs[i]},\t{testAccs[i]}")
	return 









	
