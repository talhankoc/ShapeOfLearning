import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import inf
import os
from sklearn.cluster import KMeans


# input is nparray with shape = (any,2) that represents points in form of x,y values
# returns mean, std, lifetime (y = y-x), and points whose lifetime are one standard deviation higher than mean
def analysis(data):
	lifetime = np.array([ [x,y-x] for x,y in data if y != inf ])
	mean, std = np.mean(lifetime[:,1]), np.std(lifetime[:,1])
	above_standard_deviation_points = np.array([point for point in lifetime if point[1] > mean + std ])
	return lifetime, mean, std, above_standard_deviation_points


path = 'Homology/Data/CIFAR-10-Variation2/'
file_name = 'VRFiltration_BettiData.txt'
folder_prefix = 'model-'
epochs = range(101)

for epoch in epochs:

	print('Epoch', epoch)
	folder_path = path + folder_prefix + str(epoch) +'/'
	fn = folder_path + file_name
	x_range = 25.0

	with open(fn, 'rb') as f:
		diagrams = pickle.load(f)

	data_betti1 = diagrams[1]

	kmeans = KMeans(n_clusters=2)
	kmeans.fit(data_betti1)
	y_kmeans = kmeans.predict(data_betti1)

	y_kmeans = y_kmeans if sum(y_kmeans) < len(y_kmeans)/2 else [(x+1)%2 for x in y_kmeans]# make 0's the bigger cluster
	big_cluster   = [point for point,cluster in zip(data_betti1,y_kmeans) if cluster == 0]
	small_cluster = [point for point,cluster in zip(data_betti1,y_kmeans) if cluster == 1]

	h1_lifetime_points, mean_h1_life, std_h1_life, above_standard_deviation_h1 = analysis(data_betti1)
	big_cluster_lifetime_points, mean_big_cluster, std_big_cluster, above_standard_deviation_big_cluster = analysis(big_cluster)
	small_cluster_lifetime_points, mean_small_cluster, std_small_cluster, above_standard_deviation_small_cluster = analysis(small_cluster)

	table = {\
			'mean_h1_life':mean_h1_life, 'std_h1_life':std_h1_life,\
			'mean_big_cluster':mean_big_cluster, 'std_big_cluster':std_big_cluster,\
			'mean_small_cluster':mean_small_cluster, 'std_small_cluster':std_small_cluster,\
			'big_cluster_lifetime_points':big_cluster_lifetime_points,\
			'small_cluster_lifetime_points':small_cluster_lifetime_points,\
			'above_standard_deviation_big_cluster':above_standard_deviation_big_cluster,\
			'above_standard_deviation_small_cluster':above_standard_deviation_small_cluster,\
			'above_standard_deviation_h1':above_standard_deviation_h1,\
			'above_standard_deviation_h1_mean':np.mean(above_standard_deviation_h1[:,1]),\
			'above_standard_deviation_big_cluster_mean':np.mean(above_standard_deviation_big_cluster[:,1]),\
			'above_standard_deviation_small_cluster_mean':np.mean(above_standard_deviation_small_cluster[:,1])\
			}

	with open(folder_path + 'clusterAnalysis.txt', 'wb') as analysis_file:
		pickle.dump(table, analysis_file)

	print('Betti 1 Count :', len(data_betti1), '\t\t\t\tmean life:',mean_h1_life, \
		'\tmean life of std+1: ',table['above_standard_deviation_h1_mean'] )
	print('Cluster1 H1 Portion :', float(len(big_cluster)/len(data_betti1)), '\tmean life:', mean_big_cluster,\
		'\tmean life of std+1: ',table['above_standard_deviation_big_cluster_mean'])
	print('Cluster2 H1 Portion :', float(len(small_cluster)/len(data_betti1)), '\tmean life:', mean_small_cluster,\
		'\tmean life of std+1: ',table['above_standard_deviation_small_cluster_mean'])

