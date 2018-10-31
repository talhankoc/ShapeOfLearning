import drawNetwork

epoch = [2,3,4,5,10,15,20,25,30]
test = [2,3]
layers = [4,8,12,16,24]

constant = "/Users/kunaalsharma/Desktop/MATH 493/ShapeOfLearning/Homology/"
def symbName(e, t, l):
	return "digits_"+str(e)+"_"+str(t)+"_"+str(l)

def pathName(e,t,l):
	return constant + "Variable Units - "+str(e)+" epochs/"+"Test"+str(t)+"/SimpleNN-"+str(l)

for e in epoch:
	for t in test:
		for l in layers:
			drawNetwork.main(["",constant+"BettiData/"+symbName(e,t,l)+"/",
				symbName(e,t,l), constant+"NetworkImages/",0.05])

