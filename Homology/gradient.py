import numpy as np
import os, sys, pickle, re

path = "/Users/kunaalsharma/Desktop/Math 493/ShapeOfLearning/Homology/DigitsFull/"
analysis = "/analysis.txt"

def getName(e,w):
    return "Digits_"+ str(e) + "_" + str(w)

def calculateGradient(points):
    return [np.abs(points[i]-points[i-1]) for i in range(1,len(points))]
    #return [np.abs(points[i]-(sum(points[i+1:i+numPoints+1]))/numPoints) for i in range(0,len(points)-(numPoints+1))]

def getStopping(points):
    stoppingVal = 0.005
    for i in range(0,len(points)):
        if points[i]<=stoppingVal:
            return i
    return -1

def getTestingStopping(points):
    stoppingPercent = 0.005
    numPointsToConsider = 30
    finalAccuracy = sum(points[-numPointsToConsider:])/numPointsToConsider

    for i in range(0,len(points)):
        if np.abs(finalAccuracy-points[i])<stoppingPercent:
            return i
    return -1

def getTestingData(w):
    testPath = "/Users/kunaalsharma/Desktop/Math 493/ShapeOfLearning/NNGeneration/Saved Models/Digits - Each Epoch/Scores/"

    fn_pre = 'HiddenLayerNodeCount'
    fn_post = '_Scores.txt'

    fn = testPath + fn_pre + str(w) + fn_post
    test_scores = []
    train_scores = []
    nums = []
    with open(fn, 'rb') as f:
        nums = re.findall(b"\d+\.\d+", f.read())
    for i in range(len(nums)):
        if i%2==0:
            test_scores.append(1-float(nums[i]))
        else:
            train_scores.append(1-float(nums[i]))

    return test_scores



epochs = [i for i in range(1,51)]
widths = [8,16,24,32,40,48]


for w in widths:
    b1Average = []
    for e in epochs:
        name = getName(e,w)
        currEpochData = pickle.load(open(path + name + analysis,"rb"))
        b1Average.append(currEpochData[3])

    b1AverageStopping = getStopping(calculateGradient(b1Average))
    b1TestStopping = getTestingStopping(getTestingData(w))
    print("Data for : "+getName(e,w))
    print("B1 stopping: ",b1AverageStopping," Test stopping: ",b1TestStopping)





