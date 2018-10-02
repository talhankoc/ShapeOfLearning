import numpy,sys,os
import matplotlib.pyplot as plt


'''
Computes betti 0 and betti1 for an abstract graph saved as an adjacency matrix.
argv = ["", filename] where filename is a file with a numpy saved adjacency matrix in it

'''
def main(argv):
	global boundaryMatrix
	global boundarySize
	global allLowestOnes
	allLowestOnes=[]
	if len(argv)<1:
		print("Please run with arguments\n")
		return (-1,-1)
	matrix = getAdjacencyMatrix(argv[0])
	vertices,edges,edgeList = countEdges(matrix)
	boundaryMatrix = makeBoundary(vertices,edges,edgeList)
	boundarySize = boundaryMatrix.shape[0]
	getAllLowestOnes()
	reduceMatrix()
	return computeHomology(vertices)


#return the adjacency matrix from file
def getAdjacencyMatrix(filename):
	matrix = numpy.load(filename)
	return matrix

#converts an adjacency  matrix into G = V,E
def countEdges(matrix):
	vertices = matrix.shape[0]
	edges = 0
	edgeList = []
	for i in range(0,vertices):
		for j in range(i,vertices):
			if matrix.item(i,j)==1:
				edges+=1
				edgeList.append((i,j))
	return vertices,edges,edgeList

#converts G = V,E into a boundary matrix as per Elsbrunner
def makeBoundary(vertices,edges,edgeList):
	boundaryMatrixrix = numpy.zeros((vertices+edges,vertices+edges),dtype=numpy.int)
	for num, (i,j) in enumerate(edgeList):
		index = num + vertices
		boundaryMatrixrix[i,index] = 1
		boundaryMatrixrix[j,index] = 1
	return boundaryMatrixrix

#Get all lowest ones utility function for homology computation
def getAllLowestOne():
	global allLowestOnes
	for i in range(0,boundarySize):
		allLowestOnes.append(get_low(i))


#returns the index of the lowest 1 in a column
def get_low(j):
	for i in range(boundarySize-1,-1,-1):
		if boundaryMatrix.item(i,j)==1:
			return i
	return -1

#checks to see if there is a lower 1 and returns the index
def has_lower(j):
	if j==0 or allLowestOnes[j]==-1:
		return -1
	for i in range(0,j):
		if allLowestOnes[i]==allLowestOnes[j]:
			return i
	return -1

#adds two columns modulo 2 
def addColumn(low,high):
	global boundaryMatrix
	global allLowestOnes
	for i in range(0,boundarySize):
		if boundaryMatrix.item(i,low)==1:
			if boundaryMatrix.item(i,high)==1:
				boundaryMatrix[i,high]=0
			else:
				boundaryMatrix[i,high]=1
	allLowestOnes[high] = get_low(high)

#returns a reduced boundary matrix
def reduceMatrix():
	for i in range(0,boundarySize):
		low_index = has_lower(i)
		while(low_index!=-1):
			addColumn(low_index,i)
			low_index = has_lower(i)

#returns betti 0 and betti 1
def computeHomology(vertices):
	z0,b0,z1,b1 = 0,0,0,0
	for i in range(0,vertices):
		if allLowestOnes[i]==-1:
			z0+=1
		if i in allLowestOnes:
			b0+=1
	for i in range(vertices,boundarySize):
		if allLowestOnes[i]==-1:
			z1+=1
		if i in allLowestOnes:
			b1+=1
	return (z0-b0,z1-b1)
	

if __name__=="__main__":
	main(sys.argv[1:])