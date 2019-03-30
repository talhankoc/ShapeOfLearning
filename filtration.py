from ripser import ripser, plot_dgms
import pickle

'''
Runs ripser on a matrix, and saves the results at the given location
'''
def VR(path,matrix,is_distance_matrix=True):
	ret = ripser(matrix, maxdim=1, distance_matrix=is_distance_matrix)
	diagrams = ret['dgms']
	plot_dgms(diagrams)
	with open(path, "wb") as f:
		pickle.dump(diagrams, f)