from tensorflow import keras

class TrainTest(keras.callbacks.Callback):
    def setTest(self,x,y):
       self.testX = x
       self.testY = y
       return self

    def setModel(self,model):
       self.model = model
       return self

    def on_train_begin(self, logs={}):
        self.trainAccs = []
        self.testAccs = []

    def on_epoch_end(self, epoch, logs={}):
        self.trainAccs.append(1-logs.get('loss'))
        preds = self.model.predict(self.testX)
        correctPreds = len([1 for i in range(len(preds)) if preds[i]==self.testY[i]])
        testAcc = correctPreds/len(preds)
        self.testAccs.append(testAcc)

    def returnAccs():
        return (self.trainAccs,self.testAccs)