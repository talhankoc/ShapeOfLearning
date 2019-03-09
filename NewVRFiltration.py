import sys, os
from loadweights import loadWeights
from generateAdjacencyMatrix import getWeightedAdjacencyMatrixNoBias
import numpy as np
from numpy import array, asarray, inf, zeros, minimum, diagonal, newaxis
from ripser import ripser, plot_dgms
from ripser import Rips
import pickle

savePath = "Homology/Data/CIFAR-10-AML/"
symbName = None

'''
This function goes to the file specified by path, and creates a weighted, unbiased adjacency matrix.
Then, connections between unconnected vertices are added according to the algorithm below. Then, layers
are renormalized to be between 0 and 1, and the betti numbers are computed.
'''
def main(argv):
	global symbName
	path = argv[0]
	symbName = argv[1]

	try:
		os.makedirs(savePath + symbName)
	except:
		if os.path.isfile(generateBettiSavePath()):
			print("Betti data already exists for: "+generateBettiSavePath())
			return

	print('Obtaining adjacency matrix...')
	weights = loadWeights(path,['dense_1'])
	matrix = getWeightedAdjacencyMatrixNoBias(weights)
	print('Transforming weight matrix into distance matrix...')
	matrix = makeWeightAbsoluteDistance(matrix)
	#np.save(savePath + symbName + "/" + 'originalMatrix.npy', matrix)
	matrix = removeZeros(matrix)
	print('Running Floyd-Warshall...')
	matrix = floyd_warshall_fastest(matrix)
	#np.save(savePath + symbName + "/" + 'shortMatrix.npy', matrix)
	print('Running VR Filtration...')
	runVRFiltration(matrix)
	return

'''
This function returns the location to which the shortest distance matrix will be saved
'''
def generateMatrixSavePath():
	return savePath + symbName + "/"+"shortestDistanceMatrix.npy"

'''
This function returns the location to which the computed betti numbers will be saved
'''
def generateBettiSavePath():
	return savePath + symbName + "/VRFiltration_BettiData.txt"



'''
This function takes in a matrix with entries representing weights between pairs of vertices, 
and returns a transformed matrix with each entry being 1/the weight
'''
def makeWeightAbsoluteDistance(matrix):
	for i in range(0,matrix.shape[0]):
		for j in range(0,matrix.shape[0]):
			currItem = matrix.item(i,j)
			if not currItem==0:
				matrix.itemset((i,j),abs(1/currItem))
	return matrix


def removeZeros(matrix):
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			if i != j and matrix.item(i,j) == 0:
				matrix.itemset((i,j), inf)
	return matrix


def check_and_convert_adjacency_matrix(adjacency_matrix):
    mat = asarray(adjacency_matrix)

    (nrows, ncols) = mat.shape
    assert nrows == ncols
    n = nrows

    assert (diagonal(mat) == 0.0).all()

    return (mat, n)
    
'''floyd_warshall_fastest(adjacency_matrix) -> shortest_path_distance_matrix
Input
    An NxN NumPy array describing the directed distances between N nodes.
    adjacency_matrix[i,j] = distance to travel directly from node i to node j (without passing through other nodes)
    Notes:
    * If there is no edge connecting i->j then adjacency_matrix[i,j] should be equal to numpy.inf.
    * The diagonal of adjacency_matrix should be zero.
Output
    An NxN NumPy array such that result[i,j] is the shortest distance to travel between node i and node j. If no such path exists then result[i,j] == numpy.inf
'''
def floyd_warshall_fastest(adjacency_matrix):

    (mat, n) = check_and_convert_adjacency_matrix(adjacency_matrix)

    for k in range(n):
        mat = minimum(mat, mat[newaxis,k,:] + mat[:,k,newaxis]) 

    return mat


def runVRFiltration(matrix):
	ret = ripser(matrix, maxdim=1, distance_matrix=True)
	diagrams = ret['dgms']
	with open(generateBettiSavePath(), "wb") as f:
		print("About to save diagrams...."+generateBettiSavePath())
		pickle.dump(diagrams, f)
	

if __name__ == "__main__":
	main(sys.argv[1:])