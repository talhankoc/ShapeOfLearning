import sys, makeAllGraphs, getCutoffsAndHomology, computeHomology


numVertices = 0
inputVertices = 784
outputVertices = 10
symbName = ""
path = ""
savePath = "/home/ec2-user/ShapeOfLearning/Homology/BettiData/NewMethod/"

'''
This function goes to the file specified by path, and creates a weighted, unbiased adjacency matrix.
Then, connections between unconnected vertices are added according to the algorithm below. Then, layers
are renormalized to be between 0 and 1, and the betti numbers are computed.
'''
def main(argv):
	global numVertices
	global symbName
	numVertices = int(argv[0])
	symbName = argv[1]
	path = argv[2]

	generateRawAdjacencyMatrix()
	matrix = addConnectionsToMatrix()
	matrix = renormalizeMatrixLayers(matrix)
	runBetti(matrix)
	return


'''
This function returns the location to which the raw matrix (uncomputed) will be saved
'''
def generateMatrixSavePath():
	return savePath + symbName + ".npy"

'''
This function returns the location to which the computed betti numbers will be saved
'''
def generateBettiSavePath():
	return savePath + symbName + "_BettiData.txt"

'''
This function returns the location to which the cutoff matrix should be saved
'''
def generateCutoffSavePath(cutoff):
	return savePath + symbName + str(cutoff)

'''
This function calls makeAllGraphs, and generates the raw adjacency matrix. This is
saved in the savePath location
'''
def generateRawAdjacencyMatrix():
	makeAllGraphs.main(["","0","-w","-nb",generateMatrixSavePath(),path])

'''
This function takes the matrix, and runs the connetion algorithm on it. The matrix
is then returned
'''
def addConnectionsToMatrix():
	matrix = convertWeightToDistance(computeHomology.get_adjacency_matrix(generateMatrixSavePath()))
	outputLayerEnd = matrix.shape[0]
	outputLayerStart = outputLayerEnd - 10
	
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
				connectionStrength = -1
				if (currLayerStart==inputVertices):
					connectionStrengthPrev = getConnectionForLayer(matrix,v1,v2,0,inputVertices)
					connectionStrengthNext = getConnectionForLayer(matrix,v1,v2,currLayerEnd,currLayerEnd+numVertices)
					connectionStrength = (connectionStrengthNext + connectionStrengthPrev) / 2
				else if (currLayerEnd==outputLayerStart):
					connectionStrengthPrev = getConnectionForLayer(matrix,v1,v2,currLayerStart-numVertices,currLayerStart)
					connectionStrengthNext = getConnectionForLayer(matrix,v1,v2,outputLayerStart,outputLayerEnd)
					connectionStrength = (connectionStrengthNext + connectionStrengthPrev) / 2
				else:
					connectionStrengthPrev = getConnectionForLayer(matrix,v1,v2,currLayerStart-numVertices,currLayerStart)
					connectionStrengthNext = getConnectionForLayer(matrix,v1,v2,currLayerEnd,currLayerEnd+numVertices)
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
		currConnection = matrix.item(v1,i) * matrix.item(v2,i)
		allConnections.append(currConnection)
	allConnections.sort()

	totalConnection = 0
	listStart = len(allConnections)*0.4
	listEnd = len(allConnections)*0.6
	for i in range(listStart,listEnd):
		totalConnection += allConnections[i]

	return totalConnection/(listEnd-listStart)

'''
This function takes in a matrix with entries representing weights between pairs of vertices, 
and returns a transformed matrix with each entry being 1/the weight
'''
def convertWeightToDistance(matrix):
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
	outputLayerStart = outputLayerEnd - 10
	
	renormalizeLayer(matrix,0,inputVertices,-1,-1)
	currStart = inputVertices
	currEnd = inputVertices + numVertices
	renormalizeLayer(matrix,0,inputVertices,currStart,currEnd)

	while(currEnd<=outputLayerEnd):
		renormalizeLayer(matrix,currStart,currEnd,-1,-1)
		if (currEnd==outputLayerStart):
			renormalizeLayer(matrix,currStart,currEnd,outputLayerStart,outputLayerEnd)
		else:
			renormalizeLayer(matrix,currStart,currEnd,currStart + numVertices,currEnd + numVertices)
		currStart += numVertices
		currEnd += numVertices

	renormalizeLayer(matrix,outputLayerStart,outputLayerEnd)


'''
This function normalizes a layer of a matrix. If nextStart and nextEnd are -1, 
then you normalize within the layer rather than between layers
'''
def renormalizeLayer(matrix,start,end,nextStart,nextEnd):
	currMax = -1
	if (nextStart==-1 and nextEnd==-1):
		nextStart = start + 1
		nextEnd = end 
	else:
		end = end + 1

	for v1 in range(start,end-1):
		for v2 in range(nextStart,nextEnd):
			currEdge = matrix.item(v1,v2)
			if currEdge > currMax:
				currMax = currEdge

	for v1 in range(start,end-1):
		for v2 in range(nextStart,nextEnd):
			currItem = matrix.item(v1,v2)/currMax
			matrix.itemset((v1,v2),currItem)
			matrix.itemSet((v2,v1),currItem)

'''
This function takes a matrix, and runs the betti calculations for each cutoff
'''
def runBetti(matrix):
	cutoffs = getCutoffsAndHomology.generateAllCutoffSteps(0.05,1,0.05)
	allBetti = []
	for cutoff in cutoffs:
		cutoffMatrix = getCutoffMatrix(matrix,cutoff)
		numpy.save(generateCutoffSavePath(cutoff),cutoffMatrix)
		betti0,betti1 = computeHomology.main([generateCutoffSavePath(cutoff)+".npy"])
		allBetti.append([betti0,betti1])

		print("Cutoff: "+str(cutoff)+", Betti0: "+str(betti0)+", Betti1: "+str(betti1))

	np.savetxt(generateBettiSavePath(),np.array(allBetti))
	return

'''
This function takes a matrix and a cutoff, and removes all edges lower than the cutoff
'''
def getCutoffMatrix(matrix,cutoff):
	newMatrix = numpy.zeros(matrix.shape)

	for i in range(0,matrix.shape[0]):
		for j in range(0,matrix.shape[0]):
			currItem = matrix.item(i,j)
			if (currItem>=cutoff):
				newMatrix.itemset((i,j),currItem)
			else:
				newMatrix.itemset((i,j),0)
	return newMatrix


if __name__ == "__main__":
	main(sys.argv[1:])