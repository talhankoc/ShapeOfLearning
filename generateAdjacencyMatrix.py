import numpy as np

def getWeightedAdjacencyMatrixNoBias(W):
	#print(f'\tcombining {len(W)} layers to form adjacency matrix...')
	row_dimensions = [w.shape[0] for w in W]
	dim = sum(row_dimensions) + W[-1].shape[1]
	M = np.zeros((dim,dim))
	rowOffset = 0
	colOffset = W[0].shape[0]
	placeSmallerInBiggerMatrix(rowOffset, colOffset, W[0], M)
	for i in range(1,len(W)):
		rowOffset += W[i-1].shape[0]
		colOffset += W[i].shape[0]
		placeSmallerInBiggerMatrix(rowOffset, colOffset, W[i], M)
	assert check_symmetric(M)
	return M

def placeSmallerInBiggerMatrix(rowOffset,colOffset, smaller,bigger):
	for i in range(0, smaller.shape[0]):
		for j in range(0, smaller.shape[1]):			
			bigger[i+rowOffset,j+colOffset] = smaller[i,j]
			bigger[j+colOffset, i+rowOffset] = smaller[i,j]

def check_symmetric(a, tol=1e-8):
	return np.allclose(a, a.T, atol=tol)
