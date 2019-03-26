import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from numpy import inf
import os



def run(bettiAnalysisPaths, savePath):
	if not os.path.isdir(savePath):
		os.mkdir(savePath)
	epochs = [i for i in range(1,len(bettiAnalysisPaths)+1)]
	count = []
	mean = []
	std = []
	for i, fn in enumerate(bettiAnalysisPaths):
		with open(fn, 'rb') as f:
			dic = pickle.load(f)
			count.append(dic['count'])
			mean.append(dic['mean'])
			std.append(dic['std'])
			
	plt.plot( 'Epochs', 'Mean Life', data=pd.DataFrame({'Epochs': epochs,  'Mean Life':mean}),\
		marker='o', markerfacecolor='red', markersize=3, color='magenta', linewidth=2)
	plt.legend()
	#fn = path + 'average_life_Betti1_Closeup.png'
	fn = savePath + 'CycleMeanLife.png'
	plt.savefig(fn, dpi=400)
	plt.close()

	plt.plot( 'Epochs', 'Betti Count', data=pd.DataFrame({'Epochs': epochs,  'Betti Count':count}),\
		marker='o', markerfacecolor='red', markersize=3, color='magenta', linewidth=2)
	plt.legend()
	#fn = path + 'average_life_Betti1_Closeup.png'
	fn = savePath + 'CycleCount.png'
	plt.savefig(fn, dpi=400)
	plt.close()


