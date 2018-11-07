import NewMethod, sys, time
from multiprocessing import Process

epoch = [i for i in range(1,51)]
layers = [8,16,24,32,40,48]

constant = "/home/ec2-user/ShapeOfLearning/NNGeneration/Saved Models/Fashion - Each Epoch/"

def symbName(e,l):
	return "NewFashion2_"+str(e)+"_"+str(l)

def pathName(e,l):
	return constant + "NN-Fashion-"+str(l)+"__Epoch"+str(e)

if __name__=="__main__":
	workerInputs = []
	for e in [1]:#epoch:
		for l in [8]:#layers:
			workerInputs.append([str(l),symbName(e,l),pathName(e,l)])
	argCounter = 0
	numProcesses = int(sys.argv[1])
	pool = []
	for num in range(0,numProcesses):
		pool.append(Process(target=NewMethod.main,
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



'''
How did the weights change in this same time period 
Is the drop off proportional to the 
'''

