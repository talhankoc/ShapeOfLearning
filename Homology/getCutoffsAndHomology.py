import os,sys,computeHomology,makeAllGraphs
import numpy as np
import matplotlib
matplotlib.use('agg',warn=False, force=True)
import matplotlib.pyplot as plt
import JoeMethod

'''
Generates adjacency matrices for different cutoff values and also computes betti numbers
for these matrices.

main([string inputFileName,float cutoffStep]) 

Takes one input file, 
'''

savePathUnweighted = "/home/ec2-user/ShapeOfLearning/Homology/BettiData/"
path = ""
symbolicName = ""
def main(argv):
	cutoffs = generateAllCutoffSteps(0.05,1.0,float(argv[1]))
	global path
	global symbolicName
	path = argv[2]
	symbolicName = argv[3]
	try:
		os.makedirs(savePathUnweighted+symbolicName)
		makeGraphGivenCutoff(cutoffs,path)
	except:
		print("Matrices already exist for: "+symbolicName)
	computeBetti(cutoffs,path)
	return

'''
Takes a start, end, and cutoffStep, and returns a list of all cutoffs in the given range (exclusive
of the end)
Eg: start: 0.05, end: 1.00, cutoffStep: 0.05 returns:
[0.05,0.10,0.15,0.20,0.25,...,0.95]

'''
def generateAllCutoffSteps(start, end, cutoffStep):
	return [x/1000.0 for x in range(int(start*1000),int(end*1000),int(cutoffStep*1000))]

'''
Returns the name of the file for a given graph, generated by a given cutoff
'''
def generateGraphFileName(cutoff):
	return savePathUnweighted + symbolicName + "/"+str(cutoff)+".npy"

'''
Returns the name of the file for the Betti Numbers of a given filtration
'''
def generateBettiFileName():
	return savePathUnweighted + symbolicName + "/BettiData.txt"

'''
This function will run makeAllGraphs, after loading the required numpy file into memory
'''
def makeGraphGivenCutoff(cutoffs,path):
	for cutoff in cutoffs:
		makeAllGraphs.main(["",str(cutoff),"-uw","-b",generateGraphFileName(cutoff),path])
	return

#computes the betti numbers for each file generated in the previous step
def computeBetti(cutoffs,path):
	allBettiNumbers = []
	for cutoff in cutoffs:
		betti0,betti1 = computeHomology.main([generateGraphFileName(cutoff)])
		allBettiNumbers.append([betti0,betti1])

		print("Cutoff: "+str(cutoff)+", Betti0: "+str(betti0)+", Betti1: "+str(betti1))

	np.savetxt(generateBettiFileName(),np.array(allBettiNumbers))
	return
 
if __name__=="__main__":
	main(sys.argv)
	