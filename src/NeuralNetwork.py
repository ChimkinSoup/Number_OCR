import numpy as np
import fileReading as reader
import os

class NeuralNetwork:
    def __init__(self, hiddenNeuronCount = 512, alpha = 0.005, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, weightDecay = 0.0001, patience = 5, batchSize = 512):
        self.weight1, self.bias1, self.weight2, self.bias2 = initializeParameters(hiddenNeuronCount) # Neural network params
        self.batchSize = batchSize
    
        # Adam optimizer states 
        #   m: First moment
        #   v: Second moment
        self.mWeight1 = np.zeros_like(self.weight1) 
        self.vWeight1 = np.zeros_like(self.weight1) 
        self.mBias1 = np.zeros_like(self.bias1)
        self.vBias1 = np.zeros_like(self.bias1)
        self.mWeight2 = np.zeros_like(self.weight2) 
        self.vWeight2 = np.zeros_like(self.weight2) 
        self.mBias2 = np.zeros_like(self.bias2)
        self.vBias2 = np.zeros_like(self.bias2)

        # Hyperparameters for Adam
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.time = 0
        self.weightDecay= weightDecay;

        # Early Stopping
        self.epochCount = 1
        self.patienceLeft = [3, 3] # Curr, Init
        self.bestAccuracy = 0
        self.previousParams = [self.weight1.copy(), self.bias1.copy(), self.weight2.copy(), self.bias2.copy()]



    def loadModel(self, weight1, bias1, weight2, bias2):
        self.weight1 = weight1
        self.bias1 = bias1 
        self.weight2 = weight2
        self.bias2 = bias2  
        self.time = 0


    def forwardPropagation(self, X):
        self.preAct1 = self.weight1.dot(X) + self.bias1
        self.act1 = leakyRelu(self.preAct1)
        self.preAct2 = self.weight2.dot(self.act1) + self.bias2
        self.act2 = softMax(self.preAct2)
        return self.preAct1, self.act1, self.preAct2, self.act2 

    def backwardPropagation(self, X, y):
        m = X.shape[1]
        hotY = oneHotConverter(y)

        dPreAct2 = self.act2 - hotY
        self.dWeight2  = (1.0 / m) * dPreAct2.dot(self.act1.T)
        self.dBias2 = (1.0 / m) * np.sum(dPreAct2, axis=1, keepdims=True)
        dPreAct1 = self.weight2.T.dot(dPreAct2) * leakyReluPrime(self.preAct1)
        self.dWeight1 = (1.0 / m) * dPreAct1.dot(X.T)
        self.dBias1 = (1.0 / m) * np.sum(dPreAct1, axis=1, keepdims=True)

    def updateParameters(self, learningRate):
        if learningRate:
            learningRate = learningRate
        else:
            learningRate = self.alpha
        self.time += 1

        # =-=-=-= Weight Decay =-=-=-=
        self.dWeight1 = self.dWeight1 + self.weightDecay * self.weight1
        self.dWeight2 = self.dWeight2 + self.weightDecay * self.weight2

        # =-=-=-= Updating weight1 =-=-=-=
        self.mWeight1 = self.beta1 * self.mWeight1 + (1 - self.beta1) * self.dWeight1
        self.vWeight1 = self.beta2 * self.vWeight1 + (1 - self.beta2) * (self.dWeight1 ** 2)
        mHat = self.mWeight1 / (1 - self.beta1 ** self.time)
        vHat = self.vWeight1 / (1 - self.beta2 ** self.time)
        self.weight1 -= learningRate * mHat / (np.sqrt(vHat) + self.epsilon)

        # =-=-=-= Updating bias1 =-=-=-=
        self.mBias1 = self.beta1 * self.mBias1 + (1 - self.beta1) * self.dBias1
        self.vBias1 = self.beta2 * self.vBias1 + (1 - self.beta2) * (self.dBias1 ** 2)
        m_hat = self.mBias1 / (1 - self.beta1 ** self.time)
        v_hat = self.vBias1 / (1 - self.beta2 ** self.time)
        self.bias1 -= learningRate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # =-=-=-= Updating weight2 =-=-=-=
        self.mWeight2 = self.beta1 * self.mWeight2 + (1 - self.beta1) * self.dWeight2
        self.vWeight2 = self.beta2 * self.vWeight2 + (1 - self.beta2) * (self.dWeight2 ** 2)
        m_hat = self.mWeight2 / (1 - self.beta1 ** self.time)
        v_hat = self.vWeight2 / (1 - self.beta2 ** self.time)
        self.weight2 -= learningRate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # =-=-=-= Updating bias2 =-=-=-=
        self.mBias2 = self.beta1 * self.mBias2 + (1 - self.beta1) * self.dBias2
        self.vBias2 = self.beta2 * self.vBias2 + (1 - self.beta2) * (self.dBias2 ** 2)
        m_hat = self.mBias2 / (1 - self.beta1 ** self.time)
        v_hat = self.vBias2 / (1 - self.beta2 ** self.time)
        self.bias2 -= learningRate * m_hat / (np.sqrt(v_hat) + self.epsilon)


    def train(self, XSet, ySet, XTest, yTest, learningRateDecay = 0.95):
        numberOfSamples = XSet.shape[1]
        learningRate = self.alpha * (learningRateDecay ** self.epochCount)
        
        for iteration in range(0, numberOfSamples, self.batchSize):
            XBatch = XSet[:, iteration : iteration + self.batchSize]
            yBatch = ySet[iteration : iteration + self.batchSize]
            self.forwardPropagation(XBatch)
            accuracy = getAccuracy(getPrediction(self.act2), yBatch)
            self.backwardPropagation(XBatch, yBatch)
            self.updateParameters(learningRate = learningRate)

        _, _, _, testPrediction = self.forwardPropagation(XTest)
        currAccuracy = getAccuracy(getPrediction(testPrediction), yTest)
        print("Epoch:", self.epochCount, "Accuracy:", currAccuracy)
        self.epochCount += 1
        if currAccuracy < self.bestAccuracy:
            self.patienceLeft = [self.patienceLeft[0] - 1, self.patienceLeft[1]]
        else:
            self.bestAccuracy = currAccuracy
            self.patienceLeft = [self.patienceLeft[1], self.patienceLeft[1]]
            self.previousParams = [self.weight1.copy(), self.bias1.copy(), self.weight2.copy(), self.bias2.copy()]
        
        if (self.patienceLeft[0] <= 0):
            self.weight1, self.bias1, self.weight2, self.bias2 = self.previousParams
            print("Patience ran out, reverted to epoch", self.epochCount - 3)
            print("Best accuracy: ", self.bestAccuracy, "with alpha: ", self.alpha)
            return False # Rerolls previous best model before stopping
        else:
            return True
    def loadBestModel(self):
        filename = "BestModel.npz"
        # Check if file exists in current directory, if not try parent of current script
        if not os.path.exists(filename):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(script_dir, "..", filename)
            if os.path.exists(candidate):
                filename = candidate
                
        model = np.load(filename)
        self.loadModel(model["weight1"], model["bias1"], model["weight2"], model["bias2"])
        
    def trainAugmented(self, XSet, ySet, XAug, yAug, XTest, yTest, learningRateDecay=0.95, regularCount = 60000, augmentedCount = 40000):
        # XSet/ySet is training data
        # XAug/yAug is training data (augmented)
        # XTest/yTest is testing data
        foundBestModel = False

        while not(foundBestModel):
            if XAug is not None and yAug is not None:
                XAug, yAug = reader.shuffleData(XAug, yAug)
                trainingImages = np.concatenate([XSet[:, :regularCount], XAug[:, :augmentedCount]], axis=1)
                trainingLabels = np.concatenate([ySet[:regularCount], yAug[:augmentedCount]], axis=0)
            else:
                trainingImages = XSet
                trainingLabels = ySet
            
            trainingImages, trainingLabels = reader.shuffleData(trainingImages, trainingLabels)
            foundBestModel = not(self.train(trainingImages, trainingLabels, XTest, yTest))
        print("Done")

    def saveData(self):
        np.savez("NN_Accuracy:" + str(self.bestAccuracy) + ".npz", weight1 = self.weight1, bias1 = self.bias1, weight2 = self.weight2, bias2 = self.bias2)       # NumPy binary file

def oneHotConverter(arr):
    oneHot = np.zeros((arr.size, 10))
    oneHot[np.arange(arr.size), arr] = 1
    return oneHot.T


def softMax(arr):
    exp_arr = np.exp(arr - np.max(arr, axis=0, keepdims=True))  # Numerical stability
    return exp_arr / np.sum(exp_arr, axis=0, keepdims=True)

def getPrediction(arr):
    return np.argmax(arr, 0)

def getAccuracy(predictions, Y):
    return round(100 * np.sum(predictions == Y) / Y.size, 2)

def leakyRelu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leakyReluPrime(x, alpha=0.01):
    gradient = np.ones_like(x)
    gradient[x < 0] = alpha
    return gradient 




def initializeParameters(hiddenNeuronCount):
    # Values for neural network with 512|10 neurons per layer 
    weight1 = np.random.rand(hiddenNeuronCount, 784) - 0.5
    bias1 = np.random.rand(hiddenNeuronCount, 1) - 0.5
    weight2 = np.random.rand(10, hiddenNeuronCount) - 0.5
    bias2 = np.random.rand(10, 1) - 0.5
    return weight1, bias1, weight2, bias2

cached_model = None

def evaluateNum(arr):
    global cached_model
    if cached_model is None:
        model = NeuralNetwork()
        try:
            model.loadBestModel()
            cached_model = model
        except Exception as e:
            raise e

    X = np.array(arr).reshape(-1, 1)  # (784, 1)
    _, _, _, prediction = cached_model.forwardPropagation(X)
    return int(getPrediction(prediction))  # returns 0–9

