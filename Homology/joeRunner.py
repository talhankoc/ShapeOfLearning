import getCutoffsAndHomology

epoch = [1,10,30,50]
layers = [16,48]

constant = "/Users/kunaalsharma/Desktop/MATH 493/ShapeOfLearning/NNGeneration/Saved Models/Fashion - Each Epoch/"

def symbName(e,l):
	return "JoeFasion"+str(e)+"_"+str(l)

def pathName(e,l):
	return constant + "NN-Fasion-"+str(l)+"__Epoch"+str(e)

for e in epoch:
	for l in layers:
		getCutoffsAndHomology.main(["",str(0.05),pathName(e,l),symbName(e,l)])
			