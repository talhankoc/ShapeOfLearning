import numpy as np
import sys,os, math
import matplotlib
matplotlib.use('agg',warn=False, force=True)
import matplotlib.pyplot as plt
import computeHomology, makeAllGraphs


JULIA_PATH = "/private/var/folders/61/tv6hd9dj145g16hpffs8ld880000gn/T/AppTranslocation/B0BC6F18-0A70-4FE2-842D-B096854CAE28/d/Julia-1.0.app/Contents/Resources/julia/bin/julia"
'''
Takes an adjacency matrix, and computes distances between every pair of points.
'''
def convertWeightsToDistances(adjacencyMatrix):
	numVertices = adjacencyMatrix.shape[0]
	for i in range(0,numVertices):
		for j in range(0,numVertices):
			currWeight = adjacencyMatrix.item(i,j)
			if (currWeight!=0):
				adjacencyMatrix.itemset((i,j),1.0/currWeight)
	return

'''
This function takes in the distance matrix, and a vertex as input. It then computes the shortest
path between the vertex and every point to which the vertex is not connected. This is reflected in the
updated distance matrix.
'''
def runDijkstra(distanceMatrix,vertex):
	numVertices = distanceMatrix.shape[0]
	distanceToVertex = {}

	for i in range(0,numVertices):
		if i==vertex:
			distanceToVertex[i] = 0
		else:
			distanceToVertex[i] = math.inf

	coveredPoints = {}

	while len(coveredPoints)!=numVertices:
		closestVertex = getClosestPoint(distanceToVertex,coveredPoints)
		currVertexDistance = distanceToVertex[closestVertex]
		coveredPoints[closestVertex] = True
		for i in range(0,numVertices):
			currEdge = distanceMatrix.item(closestVertex,i)
			if (currEdge!=0):
				if (currVertexDistance + currEdge < distanceToVertex[i]):
					distanceToVertex[i] = currVertexDistance + currEdge

	for currVertex in distanceToVertex:
		if currVertex == vertex:
			continue
		if distanceMatrix.item(currVertex, vertex)==0:
			distanceMatrix.itemset((currVertex,vertex),distanceToVertex[currVertex])
			distanceMatrix.itemset((vertex,currVertex),distanceToVertex[currVertex])

	return

'''
Takes a dictionary of distances as input, returns the vertex with the lowest distance as output
'''
def getClosestPoint(distanceToVertex,coveredPoints):
	minDistance = math.inf
	closestVertex = -1

	for vertex in distanceToVertex:
		if distanceToVertex[vertex]<=minDistance:
			if (vertex in coveredPoints):
				continue
			minDistance = distanceToVertex[vertex]
			closestVertex = vertex

	return closestVertex

'''
Loads in a file, and then computes the distance matrix. Returns the distance matrix
'''
def main(filepath,savepath):
	makeAllGraphs.main(["",str(0.0),"-uw","-b",savepath,filepath])
	matrix = computeHomology.get_adjacency_matrix(savepath+".npy")
	convertWeightsToDistances(matrix)
	numVertices = matrix.shape[0]

	for i in range(0,numVertices):
		for j in range(0,numVertices):
			if i==j:
				continue
			if matrix.item(i,j)==0:
				runDijkstra(matrix,i)
	return matrix

	









