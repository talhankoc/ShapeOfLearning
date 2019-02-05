import VRFiltration
import sys, time
from multiprocessing import Process

epoch = [i for i in range(10)] + [i for i in range(10,36,5)]

constant = "/Users/tkoc/Code/ShapeOfLearning/NNGeneration/Saved Models/CIFAR-10/"

def symbName(e):
	return "CIFAR-10_"+str(e)

def pathName(e):
	return constant + "MODEL_Epoch"+str(e)

if __name__=="__main__":
	workerInputs = []
	for e in epoch:
		workerInputs.append([str(1024),pathName(e),symbName(e)])
	
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
	
