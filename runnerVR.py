import NewVRFiltration as VRFiltration
import sys, time
from multiprocessing import Process

epoch = range(2,101)

constant = "NNGeneration/Saved Models/CIFAR-10-AML/"

def symbName(e):
	return "model-"+str(e)

def pathName(e):
	return constant + symbName(e) + '.hdf5'

if __name__=="__main__":
	workerInputs = []
	for e in epoch:
		workerInputs.append([pathName(e),symbName(e)])
	
	argCounter = 0
	numProcesses = int(sys.argv[1])
	pool = []
	for num in range(0,numProcesses):
		pool.append(Process(target=VRFiltration.main,
			args=(workerInputs[argCounter],)))
		argCounter += 1
		pool[num].start()
	while argCounter< len(workerInputs):
		for num in range(0,numProcesses):
			if (pool[num].is_alive()!=True):
				pool[num] = Process(target=VRFiltration.main,
					args=(workerInputs[argCounter],))
				argCounter += 1
				pool[num].start()
		time.sleep(1)
	
