from torch import exp, t, randn, mv, mm, cat, ones



class NeuralNetwork():
    def __init__(self, layerSize = []):
        self.layerSize = layerSize
        #self.theta = zeros(len(self.layerSize))
        #self.theta = random.rand(1, len(self.layerSize) - 1)
        self.theta = []
        
        for i in range(len(self.layerSize) - 1):
            #self.theta[i] = random.rand(layerSize[i] + 1, layerSize[i + 1])
            self.theta.append(randn(layerSize[i] + 1, layerSize[i + 1]))
            #self.theta.append(randn(layerSize[i + 1], layerSize[i] + 1))
            #self.theta = concatenate((self.theta, random.normal(0, 1/sqrt(self.layerSize[i]), (1, self.layerSize[i]))), axis = 0)
            #print(self.theta[i])
        #self.theta = DoubleTensor(self.theta)
            
    def getLayer(self, layer):
        return self.theta[layer]
        #return t(self.theta[layer]) 
    
    def forward(self, Input):
        self.Output = Input
        for i in range(len(self.layerSize) - 1):
            if (len(self.Output.shape) == 1):
                self.Output = 1 / (1 + exp(0 - mv(t(self.theta[i]), cat((ones(1), self.Output)))))
                #Output = 1 / (1 + exp(0 - mv(self.theta[i], cat((ones(1), Output)))))
            else:
                self.Output = 1 / (1 + exp(0 - mm(t(self.theta[i]), cat((ones(1, self.Output.shape[1]), self.Output)))))
                #Output = 1 / (1 + exp(0 - mm(self.theta[i], cat((ones(1, Output.shape[1]), Output)))))
        return self.Output
