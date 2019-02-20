import tensorflow as tf
from tensorflow import keras
import numpy as np
import loadHousing
import save_keras_model
import os

savePath = "/Users/kunaalsharma/Desktop/MATH 494/MATH 493/ShapeOfLearning/NNGeneration/Saved Models/"
trainX, trainY, testX, testY = loadHousing.loadData("/Users/kunaalsharma/Desktop/MATH 494/MATH 493/ShapeOfLearning/NNGeneration/data/data.train")

def genModel(layers):
	model = tf.keras.Sequential()
	for i in range(len(layers)):
		layer = layers[i]
		if i==len(layers)-1:
			model.add(tf.keras.layers.Dense(layer, activation=tf.nn.softmax))
		else:
			model.add(tf.keras.layers.Dense(layer, activation=tf.nn.relu))
	
	model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
	return model

def mkSaveDir(width):
	currPath = savePath + f"Adult{width}/"
	if not os.path.isdir(currPath):
		os.mkdir(currPath)
	return currPath

def getSavePath(epoch):
	return f"EPOCH_{epoch}"

def runAndSaveModel(width):
	testScores = []
	trainScores = []
	model = genModel([108,width,2])
	for e in range(1,51):
		model.fit(trainX,trainY,epochs=1)
		train_l,train_a = model.evaluate(trainX,trainY,verbose=0)
		test_l,test_a = model.evaluate(testX,testY,verbose=0)
		testScores.append(test_a)
		trainScores.append(train_a)
		save_keras_model.save_model(model,mkSaveDir(width),getSavePath(e))
	with open(mkSaveDir(width)+"scores.txt","w+") as f:
		for i in range(len(testScores)):
			f.write(f"{i+1},{trainScores[i]},{testScores[i]}\n")


if __name__=="__main__":
	for i in range(3,10):
		runAndSaveModel(2**i)


