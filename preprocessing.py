import numpy as np
from numpy import asarray, inf, minimum, diagonal, newaxis

'''
Standard VR Filtration preprocessing. Takes an adjacency matrix, and 
	1) turns it into a distance matrix
	2) sets all 0s to infinity
	3) runs Floyd Warshall to connect it
The resulting matrix is returned
'''
def standardVR(adjacencyMatrix):
	matrix = makeWeightAbsoluteDistance(adjacencyMatrix)
	matrix = removeZeros(matrix)
	matrix = floydWarshallFastest(matrix)
	return matrix

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
Sets all 0s in the matrix to infinity unless they are along the diagonal 
'''
def removeZeros(matrix):
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			if i != j and matrix.item(i,j) == 0:
				matrix.itemset((i,j), inf)
	return matrix

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
floydWarshallFastest(adjacency_matrix) -> shortest_path_distance_matrix
Input
    An NxN NumPy array describing the directed distances between N nodes.
    adjacency_matrix[i,j] = distance to travel directly from node i to node j (without passing through other nodes)
    Notes:
    * If there is no edge connecting i->j then adjacency_matrix[i,j] should be equal to numpy.inf.
    * The diagonal of adjacency_matrix should be zero.
Output
    An NxN NumPy array such that result[i,j] is the shortest distance to travel between node i and node j. If no such path exists then result[i,j] == numpy.inf
'''
def floydWarshallFastest(adjacency_matrix):
    (mat, n) = check_and_convert_adjacency_matrix(adjacency_matrix)

    for k in range(n):
        mat = minimum(mat, mat[newaxis,k,:] + mat[:,k,newaxis]) 

    return mat

'''
Helper function for floydWarshallFastest
'''
def check_and_convert_adjacency_matrix(adjacency_matrix):
    mat = asarray(adjacency_matrix)
    (nrows, ncols) = mat.shape
    assert nrows == ncols
    n = nrows
    assert (diagonal(mat) == 0.0).all()
    return (mat, n)