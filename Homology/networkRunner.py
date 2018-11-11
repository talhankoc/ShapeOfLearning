import drawNetwork

constant = "/Users/kunaalsharma/Desktop/MATH 493/ShapeOfLearning/Homology/"
epoch = [1,2,3,4,5,10,15,20,25,30]
test = [1,2,3]
layers = [4,8,12,16,24,32,48,64,96]
network = ["Digits"]

def symbName(n,e, t, l):
	return n+"_"+str(e)+"_"+str(t)+"_"+str(l)

def pathName(n,e,t,l):
	return constant + n + "/Variable Units - "+str(e)+" epochs/"+"Test"+str(t)+"/SimpleNN-"+str(l)

for n in network:
		for e in epoch:
			for t in test:
				for l in layers:
					drawNetwork.main(["",constant+"BettiData/"+symbName(n,e,t,l)+"/",
				symbName(n,e,t,l), constant+"NetworkImages/",0.05])