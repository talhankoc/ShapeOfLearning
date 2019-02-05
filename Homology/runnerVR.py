import VRFiltration as VRFiltration
import sys, time
from multiprocessing import Process

epoch = range(1,51)
epoch = [1,5,10]
layers = [8,16,24,32,40,48,56,64]
layers = [32,64]

constant = "/Users/tkoc/Code/ShapeOfLearning/NNGeneration/Saved Models/Fashion - D5 - CNN - NODROPOUT/"

def symbName(e,l):
	return "Digits_"+str(e)+"_"+str(l)

def pathName(e,l):
	return constant + "NN-Digit-"+str(l)+"__Epoch"+str(e)

if __name__=="__main__":
	workerInputs = []
	for e in epoch:
		for l in layers:
			workerInputs.append([str(l),pathName(e,l),symbName(e,l)])
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
