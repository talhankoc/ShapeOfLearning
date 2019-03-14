import makeAllGraphs
import generateAdjacencyMatrix as GAM
import loadweights

# mlist is a list of np matrices all with same length and height
def sum_difference(m1,m2):
	s = 0
	for i in range(len(m1)):
		for j in range(len(m1[0])):
			s += abs(m1[i,j] - m2[i,j])
	return s

def delta(mlist):
	ret = []
	for i in range(1, len(mlist)):
		ret.append(sum_difference(mlist[i], mlist[i-1]))
	return ret

def symbName(e):
	return "CIFAR-10_"+str(e)

def pathName(e):
	return constant + "MODEL_Epoch"+str(e)

filepath="NNGeneration/Saved Models/CIFAR-10-AML/model-{epoch:02d}.hdf5"
"NNGeneration/Saved Models/CIFAR-10-Variation2/MODEL_Epoch9_W7.npy"

W = loadweights.loadWeights(filepath.format(epoch=str(10)), ['dense_1'])
M1 = GAM.getWeightedAdjacencyMatrixNoBias(W)
M2 = makeAllGraphs.main(["","0","-w","-nb",None,path + fn + str(e)])
print(np.allclose(M1, M2, atol=tol))
def check_symmetric(a, tol=1e-8):
	return 

assert False
epochs = 125
matrices = []
for e in range(epochs+1):
	m = makeAllGraphs.main(["","0","-w","-nb",None,path + fn + str(e)])
	matrices.append(m)

ret = delta(matrices)
print(ret)
