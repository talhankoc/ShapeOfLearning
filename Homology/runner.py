import getCutoffsAndHomology

network = ["Digits","Fashion"]
epoch = [1,2,3,4,5,10,15,20,25,30]
test = [1,2,3]
layers = [4,8,12,16,24,32,48,64,96,128,256]

constant = "/home/ec2-user/ShapeOfLearning/NNGeneration/Saved Models/"

def symbName(n,e, t, l):
	return n+"_"+str(e)+"_"+str(t)+"_"+str(l)

def pathName(n,e,t,l):
	return constant + n + "/Variable Units - "+str(e)+" epochs/"+"Test"+str(t)+"/SimpleNN-"+str(l)

for n in network:
	for e in epoch:
		for t in test:
			for l in layers:
				#try:
				getCutoffsAndHomology.main(["",str(0.05),pathName(n,e,t,l),symbName(n,e,t,l)])
				#except:
					#print("Skipping "+symbName(n,e,t,l))

'''
How did the weights change in this same time period 
Is the drop off proportional to the 
'''

