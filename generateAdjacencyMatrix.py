import numpy as np

def getWeightedAdjacencyMatrixNoBias(W):
	
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

'''
Matrix takes on this format
	Input, H1, . . Hn
H1  xxxxxxxxxxxxxxxxx
.	xxxxxxxxxxxxxxxxx
.	xxxxxxxxxxxxxxxxx
Hn	xxxxxxxxxxxxxxxxx
Out	xxxxxxxxxxxxxxxxx

'''
def getRepresentationMatrix(W):
	assert len(W) > 1
	row_dim = sum([w.shape[1] for w in W])	# Hidden1 to Hidden-n and output
	col_dim = sum([w.shape[0] for w in W])	# Input to Hidden-n
	M = np.zeros((row_dim,col_dim))

	col_offset = 0
	row_offset = 0
	for w in W:
		M[row_offset:row_offset+w.shape[1],col_offset:col_offset+w.shape[0]] = w.T
		row_offset += w.shape[1]
		col_offset += w.shape[0]
	return M

def placeSmallerInBiggerMatrix(rowOffset,colOffset, smaller,bigger):
	for i in range(0, smaller.shape[0]):
		for j in range(0, smaller.shape[1]):			
			bigger[i+rowOffset,j+colOffset] = smaller[i,j]
			bigger[j+colOffset, i+rowOffset] = smaller[i,j]

def check_symmetric(a, tol=1e-8):
	return np.allclose(a, a.T, atol=tol)
