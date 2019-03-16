from loadweights import loadWeights
from generateAdjacencyMatrix import getWeightedAdjacencyMatrixNoBias as getAdjacencyMatrix

# assumes M1 is the unprocessed matrix with float('inf') for edges that are unconnected
def calculatePercentWeightsChanged(M1, M2):
	assert M1.shape == M2.shape

	rows = M1.shape[0]
	cols= M1.shape[1]
	total = 0
	count = 0
	for i in range(rows):
		for j in range(cols):
			total += 1
			if M1[i,j] == float('inf'):
				continue
			if M1[i,j] != M2[i,j]:
				count += 1

	return float(count/total)

W = loadWeights(fn)
getAdjacencyMatrix()