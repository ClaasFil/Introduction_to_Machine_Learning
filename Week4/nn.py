import numpy as np
from _stochastic_optimizers import AdamOptimizer
import copy


#activation function modular
def rbf(x):
    return 1/(1+np.exp(-x))
def ReLU(x):
    return np.maximum(0,x)



class NeuralNetwork():
    def __init__(self,layer_sizes):
        #initiate a network with a defined size
        self.layer_sizes = layer_sizes
        self.activations=[np.zeros((num_rows,1)) for num_rows in layer_sizes]
        #initiate empty biases
        self.biases=[np.zeros((num_rows,1))for num_rows in layer_sizes[1:]]
        #initiate wights as described in the lecture
        self.weight_shapes=[(a,b) for a,b in zip(layer_sizes[1:],layer_sizes[:-1])]
        self.weights=[np.random.normal(0,0.02*shape[0],size=(shape)) for shape in self.weight_shapes]
        self.num_layers = len(layer_sizes)


    def forward(self,input):
        self.activations[0]=input.reshape((input.shape[0],1))
        for l in range(1,self.num_layers):
            z=np.matmul(self.weights[l-1],self.activations[l-1])+self.biases[l-1]
            self.activations[l] = ReLU(z)

#-----getter and setter functions
    def getActivations(self):
        return self.activations

    def getBiases(self):
        return self.biases
        
    def getWeights(self):
        return self.weights

    def getOutput(self):
        return self.activations[-1]
        

    def setBiases(self, biasesIn):
        self.biases = biasesIn

    def setWeights(self, weightsIn):
        self.weights = weightsIn
#------
    def calcLoss(self, X, y, W, b):
        #define a temporary network to interfere with the current activation functions
        temp_nn=NeuralNetwork(self.layer_sizes)
        temp_nn.setWeights(W)
        temp_nn.setBiases(b)
        totalLoss = 0
        #iterate over all the datapoints
        for i, datapoint in enumerate(X):
            temp_nn.forward(datapoint)
            #MSE loss
            totalLoss += (temp_nn.getOutput()- y[i])**2  
        return float(totalLoss)

    
    def calcGrad(self, e, X, y):

        #initiate empty gradient  
        self.gradWght =[np.zeros(shape) for shape in self.weight_shapes]

        #iterate over the elements of the gradient
        for NumberLayer, layer in enumerate(self.weights):
            for (i) in range(layer.shape[0]):
                for (j) in range(layer.shape[1]):
                    #temporary copies of weight vectors
                    weightMinusE = copy.deepcopy(self.weights)
                    weightPlusE = copy.deepcopy(self.weights)

                    #modify wights to calculate one single gradient element
                    weightMinusE[NumberLayer][i,j] = weightMinusE[NumberLayer][i,j]- e
                    weightPlusE[NumberLayer][i,j] = weightPlusE[NumberLayer][i,j]+ e

                    #calc loss with modified weights
                    lossEMinus = self.calcLoss( X, y, weightMinusE, self.biases)
                    lossEPlus = self.calcLoss( X, y, weightPlusE, self.biases)

                    #save as gradient
                    self.gradWght[NumberLayer][i,j] =  (lossEPlus-lossEMinus)/(2*e)



        #initiate empty biasses 
        self.gradBias =[np.zeros((num_rows,1))for num_rows in self.layer_sizes[1:]]

        #iterate over the elements of the gradient
        for NumberLayer, layer in enumerate(self.weights):
            for (i) in range(layer.shape[0]):
                #temporary copies of bias vectors
                biasMinusE = copy.deepcopy(self.biases)
                biasPlusE = copy.deepcopy(self.biases)

                #modify wights to calculate one single bias element
                biasMinusE[NumberLayer][i] = biasMinusE[NumberLayer][i]- e
                biasPlusE[NumberLayer][i] = biasPlusE[NumberLayer][i]+ e

                #calc loss with modified weights
                lossEMinus = self.calcLoss( X, y, self.weights, biasMinusE)
                lossEPlus = self.calcLoss( X, y, self.weights, biasPlusE)

                #save as bias
                self.gradBias[NumberLayer][i] =  (lossEPlus-lossEMinus)/(2*e)


        return self.gradWght + self.gradBias
