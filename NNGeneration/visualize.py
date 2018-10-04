import numpy as np
import matplotlib.pyplot as plt
import os
import re

#ret list of triple (node_count==int, accuracy==float)
def get_score_list(filepath):
	with os.fdopen(os.open(filepath,os.O_RDONLY),'r') as f:  
		content = f.readlines()
	node_count_list = []
	accuracy_list = []

	for line in content:
		if "Nodes" in line:
			node_count_list.append(int(re.sub('[^0-9]','', line)))
		elif "Test" in line:
			offset = len('Test accuracy: ')
			accuracy_list.append(float(line[offset:-1]))
	return zip(node_count_list,accuracy_list)

# ret = node_count, epochs,scores 
def get_data():
	root_dir = "Saved Models/"
	folder_name = root_dir + "Digits/"
	file_name = "scores.txt"
	epoch_folders = sorted([f for f in os.listdir(folder_name)  if '.' not in f])
	ret = []
	#sort folder
	for epoch_f in epoch_folders:
		epoch = re.sub('[^0-9]','', epoch_f)
		test_folders = [f for f in os.listdir(folder_name+epoch_f) if '.' not in f]
		s = []
		for test_f in test_folders:
			#score_list element is a tuple (node_count, accuracy)
			score_list = get_score_list(folder_name+epoch_f+'/'+test_f+'/'+file_name)
			s.append(score_list)
		average_node_scores = [(x[0],(x[1]+y[1]+z[1])/3) for x,y,z in zip(s[0],s[1],s[2]) ]
		for node_score in average_node_scores:
			ret.append((node_score[0], epoch, node_score[1]))

	return zip(*ret)



##[node_count][epoch] = acc -> accuracy of model with node_count and epoch
score_dict = dict()
node_counts, epochs, accuracy_list = get_data()
for node_count, epoch, accuracy in zip(node_counts, epochs, accuracy_list):
	if node_count not in score_dict:
		score_dict[node_count] = dict()
	score_dict[node_count][epoch] = accuracy

plt.plot([x for (x,y) in sorted(score_dict[256].items())],[y for (x,y) in sorted(score_dict[256].items())])
plt.legend()
plt.show()
'''

X = [256,128,96,64,48,32,24,16,12,8,4]
Y = [30,25,20,15,10,5,4,3,2,1]
Z =  np.zeros((len(X), len(Y)))

for i in range(len(X)):
	for j in range(len(Y)):
		Z[i,j] = score_dict[i][j]


ax = plt.subplots(1,2, sharex=True, sharey=True)
ax[0].tripcolor(X,Y,Z)
ax[1].tricontourf(X,Y,Z, 20) # choose 20 contour levels, just to show how good its interpolation is
ax[1].plot(X,Y, 'ko ')
ax[0].plot(X,Y, 'ko ')
plt.show()


'''
