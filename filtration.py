from ripser import ripser, Rips

'''
Runs ripser on a matrix, and saves the results at the given location
'''
def VR(path,matrix):
	ret = ripser(matrix, maxdim=1, distance_matrix=True)
	diagrams = ret['dgms']
	with open(path, "wb") as f:
		pickle.dump(diagrams, f)