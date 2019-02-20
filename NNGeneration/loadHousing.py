import numpy as np
'''
Formating function used for the adult income dataset
'''
def formatHousing(line):
	intEntries = [0,2,4,10,11,12]
	strList = []
	intList = []
	lab = 1
	for i in range(len(line)):
		if i in intEntries:
			intList.append(int(line[i]))
		elif i==len(line)-1:
			if line[i]== " <=50K.":
				lab = 0
		else:
			strList.append(line[i])
	return (intList,strList,lab)
'''
Appends together rows in the two 2d arrays passed as arguments
'''
def addToData(data,newColumns):
	for i in range(len(data)):
		data[i] = data[i] + newColumns[i]
	return data
'''
Takes in a column of categorical data and returns a one hot encoding 
'''
def encoder(column):
	cats = {}
	index = 0
	for item in column:
		if item not in cats:
			cats[item] = index
			index +=1
	newData = []
	for item in column:
		newData.append(getNewLine(index,cats[item]))
	return newData
'''
Helper function to create a one hot encoding of a line
'''
def getNewLine(size,index):
	newLine = [0] * size
	newLine[index] = 1
	return newLine
'''
Loads data from a file, one hot encodes categorical features, turns categorical
labels into real values, and returns X and Y separately 
'''
def loadData(path):
	with open(path,"r+") as f:
		rawStr = f.read()
		strLines = rawStr.split("\n")
		data = [formatHousing(i.split(",")) for i in strLines]
		currData = [j[0] for j in data]
		for i in range(len(data[0][1])):
			currCol = [j[1][i] for j in data]
			encoded = encoder(currCol)
			currData = addToData(currData,encoded)
		split = int(len(currData)*0.7)
		x = np.array(currData)
		y = np.array([j[2] for j in data])
		return x[:split], y[:split],x[split:],y[split:]


