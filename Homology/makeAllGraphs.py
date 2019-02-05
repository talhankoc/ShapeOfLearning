#python makeAllGraphs.py [cutoff value] [-w for weighted matrix, -uw for unweighted] 
# [-nb for no biases, -b for biases] [optional filename prefix for saving]
import sys
import makeGraph
import makeGraphNoBias
import numpy as np

path = ""
'''
MAIN HAS BEEN MODIFIED TO RETURN THE MATRIX INSTEAD OF SAVING IT.
'''
def main(argv):
	assert len(argv) >= 4, 'ERROR: Please enter required paramters. See readme.txt ...'
	
	global path
	#essential arguments
	cutoff = float(argv[1])
	saveName = str(argv[4])
	path = argv[5]
	matrices = None
	#flags specifying format of matrix
	if argv[2] == '-w':
		if argv[3] == '-nb':
			print('Creating weighted adjacency matrix without including the biases')
			matrices = getWeightedAdjacencyMatrixNoBias()
		elif argv[3] == '-b':
			print('Creating weighted adjacency matrix, including the biases')
			matrices = getWeightedAdjacencyMatrix()
		else:
			print('ERROR: incorrect flag. Please specify -b to include biases and -nb for no biases.')
	elif argv[2] == '-uw':
		if argv[3] == '-nb':
			print('Creating graphs without including the biases')
			matrices = getUnweightedAdjacencyMatrixNoBias(cutoff)
		elif argv[3] == '-b':
			print('Creating graphs including the biases')
			matrices = getUnweightedAdjacencyMatrix(cutoff)
		else:
			print('ERROR: incorrect flag. Please specify -b to include biases and -nb for no biases.')

	else:
		print('ERROR: incorrect flag. Please specify -w for weighted and -uw for unweighted as the second parameter.')
	#save output
	#saveBinary(matrices, saveName)

	'''

	THIS NEEDS TO BE CHANGED BACK


	'''
	return matrices

def getUnweightedAdjacencyMatrixNoBias(cutoff):
	w1 = np.load(path+'_W1.npy')
	w2 = np.load(path+'_W2.npy')
	G = makeGraphNoBias.makeGraph([w1,w2], cutoff)
	print('Vertices:', len(G[0]),'\tEdges:',len(G[1]))
	return graphToAdjacencyMatrix(G)

def getUnweightedAdjacencyMatrix(cutoff):
	w1 = np.load(path+'_W1.npy')
	w2 = np.load(path+'_W2.npy')
	b1 = np.load(path+'_b1.npy')
	b2 = np.load(path+'_b2.npy')
	b1 = np.reshape(b1, (1, -1))
	b2 = np.reshape(b2, (1, -1))
	G = makeGraph.makeGraph([w1,w2],[b1,b2], cutoff)
	print('Vertices:', len(G[0]),'\tEdges:',len(G[1]))
	return graphToAdjacencyMatrix(G)

def getWeightedAdjacencyMatrix():
	
	w1 = np.load(path+'_W1.npy')
	w2 = np.load(path+'_W2.npy')
	b1 = np.load(path+'_b1.npy')
	b2 = np.load(path+'_b2.npy')
	b1 = np.reshape(b1, (1, -1))
	b2 = np.reshape(b2, (1, -1))

	dim = w1.shape[0] + 1 + w2.shape[0] + 1 + w2.shape[1]
	m = np.zeros((dim,dim))

	inputLayerOffset = w1.shape[0] # +1 for bias layer and +1 for next starting point
	hiddenLayerOffset = inputLayerOffset + 1 + w2.shape[0]
	
	#place w1
	placeSmallerInBiggerMatrix(0, inputLayerOffset + 1, w1,m)

	#place b1
	placeSmallerInBiggerMatrix(inputLayerOffset, inputLayerOffset + 1, b1,m)

	#place w2
	placeSmallerInBiggerMatrix(inputLayerOffset+1, hiddenLayerOffset + 1, w2, m)
	#place b2
	placeSmallerInBiggerMatrix(hiddenLayerOffset, hiddenLayerOffset + 1, b2, m)

	assert check_symmetric(m)
	return m

def getWeightedAdjacencyMatrixNoBias():
	w1 = np.load(path+'_W7.npy')
	w2 = np.load(path+'_W8.npy')

	dim = w1.shape[0] + w2.shape[0] + w2.shape[1]
	m = np.zeros((dim,dim))

	inputLayerOffset = w1.shape[0] # +1 for bias layer and +1 for next starting point
	hiddenLayerOffset = inputLayerOffset + w2.shape[0]

	#place w1
	placeSmallerInBiggerMatrix(0, inputLayerOffset, w1,m)

	#place w2
	placeSmallerInBiggerMatrix(inputLayerOffset, hiddenLayerOffset, w2, m)

	assert check_symmetric(m)
	return m


def placeSmallerInBiggerMatrix(rowOffset,colOffset, smaller,bigger):
	for i in range(0, smaller.shape[0]):
		for j in range(0, smaller.shape[1]):			
			bigger[i+rowOffset,j+colOffset] = smaller[i,j]
			bigger[j+colOffset, i+rowOffset] = smaller[i,j]


def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


def graphToAdjacencyMatrix(G):
	dim = len(G[0])
	print('dim =',dim)
	m = np.zeros((dim,dim))
	for edge in G[1]:
		m[edge[0]-1,edge[1]-1] = 1
		m[edge[1]-1,edge[0]-1]
	assert adjacencyErrorCheck(m), 'Error in Adjacency Matrix'
	return m

def adjacencyErrorCheck(m):
	#make sure no node is connected to self
	for i in range(m.shape[0]):
		if m[i,i] != 0:
			return False
	#785th node is bias1
	#1086th node is boas2
	#TODO check that no node from below is connected to these
	return True


def save(m, filename):
	saveBinary(m,filename)
	saveReadable(m,filename)

def saveBinary(m, filename):
	#Binary data
	np.save(filename, m)

def saveReadable(m, filename):
	#Human readable data
	np.savetxt(filename, m)

if __name__ == '__main__':
	main(sys.argv)