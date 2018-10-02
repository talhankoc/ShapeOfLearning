import numpy,os,sys,computeHomology,makeAllGraphs
import matplotlib.pyplot as plt

'''
Generates adjacency matrices for different cutoff values and also computes betti numbers
for these matrices.

main([string inputFileName,float cutoffStep]) 
'''

graphDataDirectoryName = os.getcwd()+"/GraphData/"
bettiDataDirectoryName = os.getcwd()+"/BettiData/"
filename = ""


def main(argv):
	if len(argv)<2:
		print("Need two arguments: filename (without extension), and cutoff. Aborting.\n")
		return
	global filename
	filename = argv[0]
	cutoffs = generateAllCutoffSteps(float(argv[1]))

	return

'''
Takes a start, end, and cutoffStep, and returns a list of all cutoffs in the given range (exclusive
of the end)
Eg: start: 0.05, end: 1.00, cutoffStep: 0.05 returns:
[0.05,0.10,0.15,0.20,0.25,...,0.95]

Precise to three decimal places.
'''
def generateAllCutoffSteps(start, end, cutoffStep):
	return [x/1000.0 for x in range(int(start*1000),int(end*1000),int(cutoffStep*1000))]

'''
Returns the name of the file for a given graph, generated by a given cutoff
'''
def generateGraphFileName(cutoff):
	return graphDataDirectoryName+filename+"Cutoff"+(str(cutoff).replace(".",","))+".txt"

'''
Returns the name of the file for the Betti Numbers of a given filtration
'''
def generateBettiFileName():
	return bettiDataDirectoryName+filename+".txt"

'''
This function will run makeAllGraphs, after loading the required numpy file into memory
'''
def makeGraphGivenCutoff(cutoffs):
	for cutoff in cutoffs:
		'''
		this line of code should call makeAllGraphs, make the graph with the correct specifications, 
		and then save the graph into the data folder
		'''
	return

#computes the betti numbers for each file generated in the previous step
def computeBetti(cutoffs):
	allBettiNumbers = []
	print("Computing Homology for "+filename)
	for cutoff in cutoffs:
		betti0,betti1 = computeHomology.main([generateGraphFileName(cutoff)])
		allBettiNumbers.append([betti0,betti1])

		print("Cutoff: "+str(cutoff)+", Betti0: "+str(betti0)+", Betti1: "+str(betti1))

	numpy.savetxt(generateBettiFileName,numpy.array(allBettiNumbers))
	return
 
if __name__=="__main__":
	main(sys.argv[1:])
	