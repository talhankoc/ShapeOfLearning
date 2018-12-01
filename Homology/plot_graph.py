import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

path = '/Users/tkoc/Code/ShapeOfLearning/Homology/BettiDataFloydVR/'
file_name = 'VRFiltration_BettiData.txt'
folder_prefix = 'Fashion2_'
layer_sizes = [8,16,24,32,40,48]
epochs = 50
h0_totals,h1_totals,avg_h0_lifes,avg_h1_lifes = [],[],[],[]

for layer_size in layer_sizes:
	for epoch in range(1, epochs+1):
		folder_path = path + folder_prefix + str(epoch) + '_' + str(layer_size) + '/'
		fn = folder_path + 'analysis.txt'
		with open(fn) as f:
			h0_total,h1_total,avg_h0_life,avg_h1_life = pickle.load(f)
			h0_totals.append(h0_total)
			h1_totals.append(h1_total)
			avg_h0_lifes.append(avg_h0_life)
			avg_h1_lifes.append(avg_h1_life)

x = [i for i in range(1,51)]
plt.plot(x,h1_totals)
plt.show()
