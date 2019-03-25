import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from numpy import inf



def run(path):
	#TODO


# def analysis(data):
#     lifetime = np.array([ [x,y-x] for x,y in data if y != inf ])
#     mean, std = np.mean(lifetime[:,1]), np.std(lifetime[:,1])
#     above_standard_deviation_points = np.array([point for point in lifetime if point[1] > mean + std ])
#     return len(lifetime), mean

def generate_dataframes(epochs):#layer_size
	h0_totals, h1_totals = [], []
	avg_h0_lifes, avg_h1_lifes = [], []
	lists = [h0_totals, h1_totals,avg_h0_lifes, avg_h1_lifes]
	for epoch in epochs:
		fn = path + f"{epoch}_8"
		print(fn)
		with open(fn, 'rb') as f:
			# values = h0_total, h1_total,avg_h0_life, avg_h1_life
			values = pickle.load(f)
			h0data = analysis(values[0])
			h1data = analysis(values[1])
			values = [h0data[0],h1data[0], h0data[1],h1data[1]]
			#print(values, '\n')
			for val, lst in zip(values, lists):
				lst.append(val)

	total_count=pd.DataFrame({'Epochs': epochs,  'Betti 0':h0_totals,'Betti 1':h1_totals})
	average_life=pd.DataFrame({'Epochs': epochs,  'Betti 0':avg_h0_lifes,'Betti 1':avg_h1_lifes})
	return total_count, average_life


#for layer_size in layer_sizes:
# #############	
path = 'bettiData/DigitsSimple/'
epochs = [i for i in range(1,31)]
total_count,average_life = generate_dataframes(epochs)

plt.plot( 'Epochs', 'Betti 1', data=total_count, marker='o', markerfacecolor='red', markersize=4, color='magenta', linewidth=2)
plt.legend()
#fn = path + 'total_count_Betti1_Closeup.png'
fn = path + 'total_count_Betti1.png'
plt.savefig(fn, dpi=400)
plt.close()


plt.plot( 'Epochs', 'Betti 0', data=average_life, marker='o', markerfacecolor='red', markersize=4, color='orange', linewidth=2)
plt.legend()
#fn = path + 'average_life_Betti0_Closeup.png'
fn = path + 'average_life_Betti0.png'
plt.savefig(fn, dpi=400)
plt.close()

plt.plot( 'Epochs', 'Betti 1', data=average_life, marker='o', markerfacecolor='red', markersize=4, color='magenta', linewidth=2)
plt.legend()
#fn = path + 'average_life_Betti1_Closeup.png'
fn = path + 'average_life_Betti1.png'
plt.savefig(fn, dpi=400)
plt.close()

