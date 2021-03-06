import SimpleNN as nn
import ConvNN as cnn
import os

parameters = [[512],[256],[128], [96], [64], [48], [32], [24], [16], [12], [8], [4]]
parameters = parameters[::-1]

nn.feedFashionDataset()
cnn.feedFashionDataset()
root_directory = 'Saved Models/Fashion/'


for epoch in [1,2,3,4,5,10,15,20,25,30]:
	for test in [1,2,3]:
		model_save_directory = root_directory + "Variable Units - "+str(epoch)+" epochs/Test" + str(test) + "/"
		scores_list = []
		for l in parameters:
			if not os.path.exists(model_save_directory):
				os.makedirs(model_save_directory)
			test_acc,train_acc = nn.makeAndRunModel(l, model_save_directory, e=epoch)
			scores_list.append("Nodes by Layer"+str(l)+"\nTest accuracy: " + str(test_acc) + "\n")
		f = open(model_save_directory+"scores.txt","w+")
		for s in scores_list:
			f.write(s)
		f.close()


