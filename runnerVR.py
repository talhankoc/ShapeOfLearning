import NewVRFiltration as VRFiltration
import sys, time
from multiprocessing import Process

epoch = [1,5,10,15,20,25,50,75,100]#range(29,101)
load_path = "NNGeneration/Saved Models/CIFAR-10-Variation2/"
#save_path = "Homology/Data/CIFAR-10-Variation2/"
#filtration_fn = 'VRFiltration_BettiData.txt'

def symbName(e):
	return "model-"+str(e)

def pathName(e):
	return load_path + symbName(e) + '.hdf5'

if __name__=="__main__":

	################
	# VRFiltration #
	################
	workerInputs = []
	for e in epoch:
		workerInputs.append([pathName(e),symbName(e), load_path, e])
	
	argCounter = 0
	numProcesses = int(sys.argv[1])
	pool = []
	for num in range(numProcesses):
		pool.append(Process(target=VRFiltration.main,
			args=(workerInputs[argCounter],)))
		argCounter += 1
		pool[num].start()
	while argCounter< len(workerInputs):
		for num in range(numProcesses):
			if (pool[num].is_alive()!=True):
				pool[num] = Process(target=VRFiltration.main,
					args=(workerInputs[argCounter],))
				argCounter += 1
				pool[num].start()
		time.sleep(1)
	
	##################
	# 
	##################