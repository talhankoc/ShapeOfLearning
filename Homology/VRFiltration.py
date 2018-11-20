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
savePath = "/Users/tkoc/Code/ShapeOfLearning/Homology/Betti2DataFloydVR/"
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
	generateRawAdjacencyMatrix()
	matrix = computeHomology.get_adjacency_matrix(generateMatrixSavePath())
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
This function takes the matrix, and runs the connetion algorithm on it. The matrix is then returned
'''
def addConnectionsToMatrix():
	matrix = makeWeightAbsoluteDistance(computeHomology.get_adjacency_matrix(generateMatrixSavePath()))
	os.remove(generateMatrixSavePath())
	outputLayerEnd = matrix.shape[0]
	outputLayerStart = outputLayerEnd - outputVertices
	
	#process input layer
	for v1 in range(0,inputVertices-1): #skip last vertex because it is already checked
		for v2 in range(v1+1,inputVertices):
			connectionStrength = getConnectionForLayer(matrix,v1,v2,inputVertices,inputVertices + numVertices)
			matrix.itemset((v1,v2),connectionStrength)
			matrix.itemset((v2,v1),connectionStrength)

	#process all middle layers
	currLayerStart = inputVertices
	currLayerEnd = inputVertices + numVertices

	while(currLayerEnd<=outputLayerStart):
		for v1 in range(currLayerStart,currLayerEnd-1):
			for v2 in range(v1+1,currLayerEnd):
				connectionStrengthPrev = -1
				connectionStrengthNext = -1

				if (currLayerStart==inputVertices):
					connectionStrengthPrev = getConnectionForLayer(matrix,v1,v2,0,inputVertices)
				else:
					connectionStrengthPrev = getConnectionForLayer(matrix,v1,v2,v1 - numVertices,v1)

				if (currLayerEnd==outputLayerStart):
					connectionStrengthNext = getConnectionForLayer(matrix,v1,v2,outputLayerStart,outputLayerEnd)
				else:
					connectionStrengthNext = getConnectionForLayer(matrix,v1,v2,v2,v2 + numVertices)

				connectionStrength = (connectionStrengthNext + connectionStrengthPrev) / 2 

				matrix.itemset((v1,v2),connectionStrength)
				matrix.itemset((v2,v1),connectionStrength)

		currLayerStart += numVertices
		currLayerEnd += numVertices

	#process output layer
	for v1 in range(outputLayerStart,outputLayerEnd-1):
		for v2 in range(v1+1,outputLayerEnd):
			connectionStrength = getConnectionForLayer(matrix,v1,v2,outputLayerStart - numVertices, outputLayerStart)
			matrix.itemset((v1,v2),connectionStrength)
			matrix.itemset((v2,v1),connectionStrength)

	return matrix

'''
This function takes each vertex in the layer, and calculates the connection strength.
The total connection strength is then averaged for 40%-60%, and returned.
'''
def getConnectionForLayer(matrix,v1,v2,start,end):
	allConnections = []

	for i in range(start,end):
		currConnection = matrix.item(v1,i) + matrix.item(v2,i)
		allConnections.append(currConnection)
	allConnections.sort()

	totalConnection = 0
	listStart = int(len(allConnections)*0.4)
	listEnd = int(len(allConnections)*0.6)
	for i in range(listStart,listEnd):
		totalConnection += allConnections[i]

	return totalConnection/(listEnd-listStart)

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

'''
This function takes in a matrix, and normalizes each layer to be between 0 and 1. The 
renormalized matrix is returned
'''
def renormalizeMatrixLayers(matrix):
	outputLayerEnd = matrix.shape[0]
	outputLayerStart = outputLayerEnd - outputVertices
	
	renormalizeLayer(matrix,0,inputVertices,-1,-1)
	currStart = inputVertices
	currEnd = inputVertices + numVertices
	renormalizeLayer(matrix,0,inputVertices,currStart,currEnd)

	while(currEnd<=outputLayerStart):
		renormalizeLayer(matrix,currStart,currEnd,-1,-1)
		if (currEnd==outputLayerStart):
			renormalizeLayer(matrix,currStart,currEnd,outputLayerStart,outputLayerEnd)
		else:
			renormalizeLayer(matrix,currStart,currEnd,currStart + numVertices,currEnd + numVertices)
		currStart += numVertices
		currEnd += numVertices

	renormalizeLayer(matrix,outputLayerStart,outputLayerEnd, -1, -1)

	return matrix

'''
This function normalizes a layer of a matrix. If nextStart and nextEnd are -1, 
then you normalize within the layer rather than between layers
'''
def renormalizeLayer(matrix,start,end,nextStart,nextEnd):
	currMax = -1
	if (nextStart==-1 and nextEnd==-1):
		nextStart = start
		nextEnd = end 

	for v1 in range(start,end):
		for v2 in range(nextStart,nextEnd):
			currEdge = matrix.item(v1,v2)
			if currEdge > currMax:
				currMax = currEdge

	if start==nextStart and end==nextEnd:
		for v1 in range(start,end):
			for v2 in range(start,end):
				matrix.itemset((v1,v2),matrix.item(v1,v2)/currMax)
	else:
		for v1 in range(start,end):
			for v2 in range(nextStart,nextEnd):
				currItem = matrix.item(v1,v2)/currMax
				matrix.itemset((v1,v2),currItem)
				matrix.itemset((v2,v1),currItem)

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

def floyd_warshall_fastest(adjacency_matrix):
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
    (mat, n) = check_and_convert_adjacency_matrix(adjacency_matrix)

    for k in range(n):
        mat = minimum(mat, mat[newaxis,k,:] + mat[:,k,newaxis]) 

    return mat


def runVRFiltration(matrix):
	ret = ripser(matrix, maxdim=2, distance_matrix=True)
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