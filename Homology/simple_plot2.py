import matplotlib.pyplot as plt
import pandas as pd


path = '/Users/tkoc/Code/ShapeOfLearning/NNGeneration/Saved Models/Fashion - Each Epoch/'
fn_pre = 'HiddenLayerNodeCount'
fn_post = '_Scores.txt'
layers = [8,16,24,32,40,48]
for layer in layers:
	fn = path + fn_pre + str(layer) + fn_post
	test_scores = []
	train_scores = []
	print(fn)
	with open(fn, 'rb') as f:
		for line in f:
			l = line.split()
			if l[0] == 'Epoch':
				continue
			test_scores.append(float(l[1]))
			train_scores.append(float(l[2]))
	print(test_scores)
	data = pd.DataFrame({'Epochs': range(1,51), 'Train':train_scores, 'Test':test_scores})
	plt.plot( 'Epochs', 'Train', data=data, marker='o', markerfacecolor='red', markersize=4, color='orange', linewidth=2)
	plt.plot( 'Epochs', 'Test', data=data, marker='o', markerfacecolor='green', markersize=4, color='orange', linewidth=2)
	plt.legend()
	plt.savefig(fn + '.png', dpi=400)
	plt.close()			