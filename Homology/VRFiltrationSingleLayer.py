import sys, makeAllGraphs, getCutoffsAndHomology, computeHomology, os
import numpy as np
from numpy import array, asarray, inf, zeros, minimum, diagonal, newaxis
from ripser import ripser, plot_dgms
from ripser import Rips
import pickle


numVertices = -1
inputVertices = 784
outputVertices = 10
path = ""
savePath = "/Users/tkoc/Code/ShapeOfLearning/Homology/FloydVRInputLayer/"
symbName = ""

'''
This function goes to the file specified by path, and creates a weighted, unbiased adjacency matrix.
Then, connections between unconnected vertices are added according to the algorithm below. Then, layers
are renormalized to be between 0 and 1, and the betti numbers are computed.
'''
def main(argv):
	global numVertices
	global symbName
	global path
	numVertices = int(argv[0])
	path = argv[1]
	symbName = argv[2]

	try:
		os.makedirs(savePath + symbName)
	except:
		if os.path.isfile(generateBettiSavePath()):
			print("Betti data already exists for: "+symbName)
			return

	print('Obtaining adjacency matrix...')
	matrix = generateLayerMatrix(path + '_W1.npy')
	print('Transforming weight matrix into distance matrix...')
	matrix = makeWeightAbsoluteDistance(matrix)
	#matrix = makeWeightDistance(matrix)
	matrix = removeZeros(matrix)
	print('Running Floyd-Warshall...')
	matrix = floyd_warshall_fastest(matrix)
	print('Running VR Filtration...')
	runVRFiltration(matrix)
	return

'''
This function returns the location to which the raw matrix (uncomputed) will be saved
'''
def generateMatrixSavePath():
	return savePath + symbName + "/"+"savedMatrix.npy"

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
This function returns the location to which the cutoff matrix should be saved
'''
def generateCutoffSavePath(cutoff):
	return savePath + symbName + "/temp"+str(cutoff)

'''
This function calls makeAllGraphs, and generates the raw adjacency matrix. This is saved in the savePath location
'''
def generateRawAdjacencyMatrix():
	makeAllGraphs.main(["","0","-w","-nb",generateMatrixSavePath(),path])

'''
Function to load the second layer of the matrix with given folder path
'''
def generateLayerMatrix(path):
	matrix = np.load(path)
	dim = matrix.shape[0] + matrix.shape[1]
	ret = np.zeros((dim,dim))
	hiddenNodes = matrix.shape[0]
	outputNodes = matrix.shape[1]
	#put m into ret
	#wall[x:x+block.shape[0], y:y+block.shape[1]] = block
	ret[0:hiddenNodes, hiddenNodes:] = matrix
	matrix = matrix.T
	#put m into ret
	ret[hiddenNodes:, 0:hiddenNodes] = matrix
	#print(ret)
	return ret 
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

'''
This function takes in a matrix with entries representing weights between pairs of vertices, 
and returns a transformed matrix with each entry being 1/the weight
'''
def makeWeightDistance(matrix):
	for i in range(0,matrix.shape[0]):
		for j in range(0,matrix.shape[0]):
			currItem = matrix.item(i,j)
			if not currItem==0:
				matrix.itemset((i,j),1/currItem)
	return matrix


def removeZeros(matrix):
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			if i != j and matrix.item(i,j) == 0:
				matrix.itemset((i,j), inf)
	return matrix

def FloydWarshall(matrix):
	numVertices = matrix.shape[0]
	for k in range(0,numVertices):
		for i in range(0,numVertices):
			for j in range(0,numVertices):
				if matrix.item(i,j) > matrix.item(i,k) + matrix.item(k,j): 
					matrix.itemset((i,j),matrix.item(i,k) + matrix.item(k,j))
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
	'''
	print(ret.keys())
	print(ret['num_edges'])
	print('***Size of dgms\t',len(dgms[0]), len(dgms[1]))
	print(dgms[0])
	print(dgms[0,0])
	print(dgms[0,1].size)
	print(type(dgms[0,2]))
	'''
	plot_dgms(diagrams, show=True)
	with open(generateBettiSavePath(), "wb") as f:
		pickle.dump(diagrams, f)

if __name__ == "__main__":
	main(sys.argv[1:])