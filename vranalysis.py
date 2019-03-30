import find_xyrange
import plotBettiCurve 
import pickle
from numpy import inf
import numpy as np
from BettiPlotter import plot_dgms


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

	
