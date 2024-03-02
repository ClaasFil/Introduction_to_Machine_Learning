import numpy as np
import copy

class MiniBatch():
    def __init__(self,x , y):
        self.x = x
        self.y = y

    def getBatch(self, outputSize):
        combine = copy.deepcopy(np.append(self.x, self.y, axis=1))
        combine = np.take(combine,np.random.permutation(combine.shape[0]),axis=0,out=combine);
        X_train = combine[:outputSize, :combine.shape[1] - 1]
        y_train = combine[:outputSize, combine.shape[1] - 1]
        return X_train, y_train