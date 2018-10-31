import numpy as np
import matplotlib.pyplot as plt
import sys, getCutoffsAndHomology



path = ""
symbolicName = ""
externalPath = ""
constantFileName = "BettiData.txt"
'''
Goes to the symbolic folder of a particular graph, reads in b0 and b1, plots them as graphs, 
and saves the plots in the symbolic folder.

inputs: argv[1] = path, argv[2] = symbolicName, argv[4] = cutoff
'''

def main(argv):
	cutoffs = getCutoffsAndHomology.generateAllCutoffSteps(0.05,1.0,float(argv[4]))
	global path
	global symbolicName
	global externalPath
	path = argv[1]
	symbolicName = argv[2]
	externalPath = argv[3]
	(betti0,betti1) = getBettiData()
	drawAndSaveGraph(cutoffs,betti0,True,ylabel="betti0",external=True)
	drawAndSaveGraph(cutoffs,betti1,False,ylabel="betti1",external=True)
	return

'''
This function goes to a file defined by the path + the constant file name, 
and loads in a BettiData file saved by getCutoffsAndHomology.py. Then, 
it returns (betti0,betti1)
'''
def getBettiData():
	bettiData = np.loadtxt(path + constantFileName)
	return (bettiData[:,0],bettiData[:,1])

'''
Takes some data, plots it on a graph, and saves the graph with the given filename.
Uses default values to make things a bit faster (if you're using it for betti)
'''
def drawAndSaveGraph(x,y,b0,xlabel="cutoff",ylabel="betti",fileName=None,external=True):
	plt.plot(x,y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	if not fileName:
		plt.savefig(path+ ylabel+".png")
		if external:
			if b0:
				plt.savefig(externalPath + "Betti0 "+symbolicName + ".png")
			else:
				plt.savefig(externalPath + "Betti1 "+symbolicName + ".png")
	else:
		plt.savefig(path+ fileName)
		if external:
			if b0:
				plt.savefig(externalPath + "Betti0 "+symbolicName + ".png")
			else:
				plt.savefig(externalPath + "Betti1 "+symbolicName + ".png")
	plt.close()
	return


if __name__ == "__main__":
	main(sys.argv)