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
    printProgressBar(0, n, prefix = 'FloyWarshall:', suffix = 'Complete', length = 50)
    for k in range(n):
        mat = minimum(mat, mat[newaxis,k,:] + mat[:,k,newaxis]) 
        printProgressBar(k+1, n, prefix = 'FloyWarshall:', suffix = 'Complete', length = 50)
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

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

