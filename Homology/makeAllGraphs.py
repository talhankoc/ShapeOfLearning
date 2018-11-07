import sys
import makeGraph
import makeGraphNoBias
import numpy as np

global path

def main(argv):
	assert len(argv) >= 4, 'ERROR: Please enter required paramters. See readme.txt ...'
	cutoff = float(argv[1])
	global path
	path = argv[5]
	saveName = None

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
			matrices = nobiases(cutoff)
		elif argv[3] == '-b':
			print('Creating graphs including the biases')
			matrices = withbiases(cutoff)
		else:
			print('ERROR: incorrect flag. Please specify -b to include biases and -nb for no biases.')

	else:
		print('ERROR: incorrect flag. Please specify -w for weighted and -uw for unweighted as the second parameter.')

	

	if len(argv) > 4:
		saveName = str(argv[4])
	else:
		print('No filename was given for saving. Adjacency matrix will not be saved.')
	
	if saveName != None:
		saveBinary(matrices, saveName)

def withbiases(cutoff):
	w1 = np.load(path+'_W1.npy')
	w2 = np.load(path+'_W2.npy')
	b1 = np.load(path+'_b1.npy')
	b2 = np.load(path+'_b2.npy')
	b1 = np.reshape(b1, (1, -1))
	b2 = np.reshape(b2, (1, -1))
	G = makeGraph.makeGraph([w1,w2],[b1,b2], cutoff)
	print('Vertices:', len(G[0]),'\tEdges:',len(G[1]))	
	return graphToAdjacencyMatrix(G)

def graphToAdjacencyMatrix(G):
	dim = len(G[0])
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
	np.savetxt(filename+'.txt', m)

if __name__ == '__main__':
	main(sys.argv)