import numpy as np
import pickle, os, sys


'''
This file calculates the area of an ellipse as measured by the distribution of 
cycle points on a VR filtration graph
'''

def loadCycleData(filename):
	with open(filename,"rb") as f:
		betti1 = pickle.load(f)[1]
		return betti1

def calculateSD(betti1):
	birth = [x[0] for x in betti1]
	death = [x[1] for x in betti1]

	birthMean = sum(birth)/len(birth)
	deathMean = sum(death)/len(death)

	birthSD = (sum([(x-birthMean)**2 for x in birth])/len(birth))**0.5
	deathSD = (sum([(x-deathMean)**2 for x in death])/len(death))**0.5

	return birthSD,deathSD

def ellipseArea(birthSD,deathSD):
	return np.pi * birthSD * deathSD

def main(pathToFile):
	betti1 = loadCycleData(pathToFile)
	birthSD,deathSD = calculateSD(betti1)
	elipseArea = ellipseArea(birthSD,deathSD)
	print("Area: ",ellipseArea)
	return


if __name__=="__main__":
	if (len(sys.argv)<2):
		print("Please provide a file path")
		return
	main(sys.argv[1])




