import matplotlib.pyplot as plt
import pandas as pd
import re
import pickle

path = 'Homology/Data/CIFAR-10-AML/'

with open(path + 'scores.txt', 'r') as f:
	import re
	dic = {}
	dic['loss'] = []
	dic['acc'] = []
	dic['val_loss'] = []
	dic['val_acc'] = []
	for line in f:  #Line is a string
		#split the string on whitespace, return a list of numbers 
		# (as strings)
		numbers_str = line.split()
		#convert numbers to floats
		numbers_float = [float(x) for x in numbers_str if re.match("^\d+?\.\d+?$", x) is not None]
		dic['loss'].append(numbers_float[0])
		dic['acc'].append(numbers_float[1])
		dic['val_loss'].append(numbers_float[2])
		dic['val_acc'].append(numbers_float[3])

data = pd.DataFrame({'Epochs': range(1,126), 'Train_acc':dic['acc'], 'Test_acc':dic['val_acc']})
plt.plot( 'Epochs', 'Train_acc', data=data, marker='o', markerfacecolor='red', markersize=4, color='orange', linewidth=2)
plt.plot( 'Epochs', 'Test_acc', data=data, marker='o', markerfacecolor='green', markersize=4, color='orange', linewidth=2)
axes = plt.gca()
plt.legend()
save_name = 'Graphs/accuracy.png'
plt.savefig(path + save_name, dpi=400)
plt.close()			
