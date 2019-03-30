import tensorflow as tf
import NNGeneration.save_keras_model as saver
from tensorflow import keras 
from NNGeneration.nnUtil import randomUpdateModel
'''
Config is a dictionary passed by master.py 

Generates a simple NN defined by layerWidths, and trains
on MNIST for "epochs" epochs. Model is saved at each epoch, 
and accuracies are saved at the end.
'''
def makeAndRun(config):
	epochs = config["epochs"]
	nnSaveFn = config["nnSaveFn"] 
	accSaveFn = config["accSaveFn"]

	mnist = tf.keras.datasets.mnist
	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0
	
	model = emptyModel(config)

	train_accuracies = []
	test_accuracies = []
	for epoch in epochs:
		model.fit(x_train,y_train,verbose=1)
		train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
		test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
		train_accuracies.append(train_acc)
		test_accuracies.append(test_acc)
		saver.save_model(model,nnSaveFn(epoch),"")
		
	#save the train and test accuracy at the end 
	saveAccuracies(accSaveFn(),train_accuracies,test_accuracies)
	return

'''
Returns an untrained model without any initialized parameters. 
'''
def emptyModel(config):
	node_count_list = config["layerWidths"]
	model = keras.models.Sequential()
	model.add(keras.layers.Flatten(input_shape=(28,28)))
	for number_of_units in node_count_list:
		model.add(keras.layers.Dense(number_of_units, activation=tf.nn.relu))
	model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
	model.compile(optimizer='adam',
	          loss='sparse_categorical_crossentropy',
	          metrics=['accuracy'])
	return model

'''
Trains a model that does random updates.
'''
def randomModel(config):
	model = randomUpdateModel(emptyModel(config))
	epochs = config["epochs"]
	nnSaveFn = config["nnSaveFn"] 
	accSaveFn = config["accSaveFn"]

	mnist = tf.keras.datasets.mnist
	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0

	train_accuracies = []
	test_accuracies = []
	for epoch in epochs:
		model = randomUpdateModel(model)
		train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
		test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
		train_accuracies.append(train_acc)
		test_accuracies.append(test_acc)
		print(f"Finished update {epoch}. Train: {train_acc}, Test:{test_acc}")
		saver.save_model(model,nnSaveFn(epoch),"")

	saveAccuracies(accSaveFn(),train_accuracies,test_accuracies)
	return

def saveAccuracies(path,train_acc,test_acc):
	with open(path,"w") as f:
		for epoch in range(len(test_acc)):
			train,test = train_acc[epoch], test_acc[epoch]
			if epoch == 0:
				f.write(f"1,\t{train},\t{test}")
			else:
				f.write(f"\n{epoch+1},\t{train},\t{test}")
	return 
