import pickle
from ripser import plot_dgms

path = '/Users/tkoc/Code/ShapeOfLearning/Homology/BettiDataVR/Fashion2_2_32/VRFiltration_BettiData.txt'
with open(path, 'rb') as f:
	diagrams = pickle.load(f)
	print(diagrams)
	plot_dgms(diagrams, show=True)

