import getCutoffsAndHomology, sys, time
from multiprocessing import Process

network = ["Digits","Fashion"]
epoch = [1,2,3,4,5,10,15,20,25,30]
test = [1,2,3]
layers = [4,8,12,16,24,32,48,64,96]

constant = "/home/ec2-user/ShapeOfLearning/NNGeneration/Saved Models/"

def symbName(n,e, t, l):
	return n+"_"+str(e)+"_"+str(t)+"_"+str(l)

def pathName(n,e,t,l):
	return constant + n + "/Variable Units - "+str(e)+" epochs/"+"Test"+str(t)+"/SimpleNN-"+str(l)


if __name__=="__main__":
	workerInputs = []
	for n in network:
		for e in epoch:
			for t in test:
				for l in layers:
					workerInputs.append(["",str(0.05),pathName(n,e,t,l),symbName(n,e,t,l)])
	argCounter = 0
	numProcesses = int(sys.argv[1])
	pool = []
	for num in range(0,numProcesses):
		pool.append(Process(target=getCutoffsAndHomology.main,
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

