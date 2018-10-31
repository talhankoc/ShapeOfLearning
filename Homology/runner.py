import getCutoffsAndHomology

epoch = [1,2,3,4,5,10,15,20,25,30]
test = [1,2,3]
layers = [4,8,12,16,24,32,64]

constant = "/Users/kunaalsharma/Desktop/MATH 493/ShapeOfLearning/NNGeneration/Saved Models/Digits/"
def symbName(e, t, l):
	return "digits_"+str(e)+"_"+str(t)+"_"+str(l)

def pathName(e,t,l):
	return constant + "Variable Units - "+str(e)+" epochs/"+"Test"+str(t)+"/SimpleNN-"+str(l)

for e in epoch:
	for t in test:
		for l in layers:
			try:
				getCutoffsAndHomology.main(["",str(0.05),pathName(e,t,l),symbName(e,t,l)])
			except:
				print("Skipping "+symbName(e,t,l))

'''
How did the weights change in this same time period 
Is the drop off proportional to the 
'''

