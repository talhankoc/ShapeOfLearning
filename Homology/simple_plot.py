import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

path = '/Users/tkoc/Code/ShapeOfLearning/Homology/Data/CIFAR-10-Variation2/'
file_name = 'analysis.txt'
folder_prefix = 'CIFAR-10_'
epochs = 100

def generate_dataframes():#layer_size
	h0_totals, h1_totals, h2_totals = [], [], []
	avg_h0_lifes, avg_h1_lifes, avg_h2_lifes = [],[],[]
	lists = [h0_totals, h1_totals,avg_h0_lifes, avg_h1_lifes]
	for epoch in range(epochs+1):
		folder_path = path + folder_prefix + str(epoch) + '/'
		fn = folder_path + file_name
		with open(fn, 'rb') as f:
			# values = h0_total, h1_total, h2_total, avg_h0_life, avg_h1_life, avg_h2_lifes
			values = pickle.load(f)
			#print(values, '\n')
			for val, lst in zip(values, lists):
				lst.append(val)
	total_count=pd.DataFrame({'Epochs': range(epochs+1),  'Betti 0':h0_totals,'Betti 1':h1_totals})
	average_life=pd.DataFrame({'Epochs': range(epochs+1),  'Betti 0':avg_h0_lifes,'Betti 1':avg_h1_lifes})
	return total_count, average_life


#for layer_size in layer_sizes:
# #############	
total_count,average_life = generate_dataframes()
plt.plot( 'Epochs', 'Betti 0', data=total_count, marker='o', markerfacecolor='red', markersize=4, color='orange', linewidth=2)
#plt.plot( 'Epochs', 'Betti 2', data=total_count, marker='o', markerfacecolor='red', markersize=4, color='black', linewidth=2)
plt.legend()
#fn = path + 'total_count_' + str(layer_size) + '.png'
fn = path + 'total_count_Betti0.png'
plt.savefig(fn, dpi=400)
plt.close()

total_count,average_life = generate_dataframes()
plt.plot( 'Epochs', 'Betti 1', data=total_count, marker='o', markerfacecolor='red', markersize=4, color='magenta', linewidth=2)
#plt.plot( 'Epochs', 'Betti 2', data=total_count, marker='o', markerfacecolor='red', markersize=4, color='black', linewidth=2)
plt.legend()
#fn = path + 'total_count_' + str(layer_size) + '.png'
fn = path + 'total_count_Betti1.png'
plt.savefig(fn, dpi=400)
plt.close()


plt.plot( 'Epochs', 'Betti 0', data=average_life, marker='o', markerfacecolor='red', markersize=4, color='orange', linewidth=2)
#plt.plot( 'Epochs', 'Betti 2', data=average_life, marker='o', markerfacecolor='red', markersize=4, color='black', linewidth=2)
plt.legend()
#fn = path + 'average_life_' + str(layer_size) + '.png'
fn = path + 'average_life_Betti0.png'
plt.savefig(fn, dpi=400)
plt.close()

plt.plot( 'Epochs', 'Betti 1', data=average_life, marker='o', markerfacecolor='red', markersize=4, color='magenta', linewidth=2)
#plt.plot( 'Epochs', 'Betti 2', data=average_life, marker='o', markerfacecolor='red', markersize=4, color='black', linewidth=2)
plt.legend()
#fn = path + 'average_life_' + str(layer_size) + '.png'
fn = path + 'average_life_Betti1.png'
plt.savefig(fn, dpi=400)
plt.close()

