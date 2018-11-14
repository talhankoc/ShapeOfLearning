import VRFiltration, sys, time
from multiprocessing import Process

epoch = [i for i in range(1,51)]
layers = [8,16,24,32,40,48]

constant = "/home/ec2-user/ShapeOfLearning/NNGeneration/Saved Models/Fashion - Each Epoch/"

def symbName(e,l):
	return "Fashion2_"+str(e)+"_"+str(l)

def pathName(e,l):
	return constant + "NN-Fashion-"+str(l)+"__Epoch"+str(e)

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
				pool[num] = Process(target=getCutoffsAndHomology.main,
					args=(workerInputs[argCounter],))
				argCounter += 1
				pool[num].start()
		time.sleep(1)