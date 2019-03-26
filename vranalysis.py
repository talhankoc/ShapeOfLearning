import find_xyrange
import plotBettiCurve 
import pickle
from numpy import inf
import numpy as np
from BettiPlotter import plot_dgms


def runAnalysisAndVisualization(load_paths, savepath):

	#find range to keep diagrams same scale
	max_x = find_xyrange.run(load_paths)
	bettiAnalysisPaths = {'betti0':[], 'betti1':[]}
	#generate betti diagram for each load_path
	for i,load_path in enumerate(load_paths):
		with open(load_path, 'rb') as f:
			diagrams = pickle.load(f)
		generateBettiDiagram(diagrams, savepath, i+1, max_x)
		fn0 = f'{savepath}Epoch{i+1}_betti0DistributionAnalysis'
		fn1 = f'{savepath}Epoch{i+1}_betti1DistributionAnalysis'
		bettiDistributionAnalysis(diagrams[0], fn0)
		bettiDistributionAnalysis(diagrams[1], fn1)
		bettiAnalysisPaths['betti0'].append(fn0)
		bettiAnalysisPaths['betti1'].append(fn1)
	plotBettiCurve.run(bettiAnalysisPaths['betti0'], f'{savepath}Betti0Curves/')
	plotBettiCurve.run(bettiAnalysisPaths['betti1'], f'{savepath}Betti1Curves/')


def generateBettiDiagram(diagrams, root_save_path, i, x_range=None, plot_lifetime_too=False):
	
	
	plot_dgms(diagrams, 
			size=12,
			title=f'CIFAR-100 | Epoch {i}',
			save_path=root_save_path+f'Epoch{i}_birth_death.png',
			xy_range=[-1,x_range,-1,x_range] if x_range else None
			)
	
	if plot_lifetime_too:
		plot_dgms(diagrams, 
			size=12, 
			title=f'CIFAR-100 | Epoch {epoch}',
			save_path=root_save_path+f'Epoch{i}_lifetime.png', 
			xy_range=[-1,x_range,-1,x_range] if x_range else None, 
			lifetime=True
			)

def bettiDistributionAnalysis(datapoints, saveFn):
	lifetime = np.array([ [x,y-x] for x,y in datapoints if y != inf ])
	mean, std = np.mean(lifetime[:,1]), np.std(lifetime[:,1])
	#above_standard_deviation_points = np.array([point for point in lifetime if point[1] > mean + std ])
	with open(saveFn, 'wb') as f:
		pickle.dump(\
			{'mean':mean, 'std':std, 'count':len(lifetime)},\
			f)

	
