from torch import exp, t, randn, mv, mm, cat, ones, mean

class NeuralNetwork():
    def __init__(self, layerSize = []):
        self.Theta = []
        self.dE_dTheta = []
        self.layerSize = layerSize
        self.layerNum = len(self.layerSize)
        
        for i in range(self.layerNum - 1):
            self.Theta.append(randn(layerSize[i] + 1, layerSize[i + 1]))
            
            
    def getLayer(self, layer):
        return self.Theta[layer]
        
    
    def forward(self, Input):
        self.Input = Input
        self.Output = []
        for i in range(self.layerNum - 1):
            if (len(self.Input.shape) == 1):
                self.Output = self.Output.append(1 / (1 + exp(0 - mv(t(self.Theta[i]), cat((ones(1), self.Input))))))
                self.Input = self.Output[-1]
            else:
                self.Output = self.Output.append(1 / (1 + exp(0 - mm(t(self.Theta[i]), cat((ones(1, self.Input.shape[1]), self.Input))))))
                self.Input = self.Output[-1]
                
        return self.Output[-1]
    
    def backward(self, target):
        self.target = target
        self.Delta_last = (self.Output[-1] - self.target) * (self.Output[-1] * (1 - self.Output[-1]))
        self.Delta = []
        self.Delta = self.Delta.append(self.Delta_last)
        for i in range(self.layerNum - 1):
            if (len(self.target.shape) == 1):
                self.dE_dTheta = self.dE_dTheta.append(self.Output[self.layerNum-2-i] @ t(self.Delta[i]))
                self.sig_d = (self.Output[self.layerNum - 3 - i]) @ (1 - self.Output[self.layerNum - 3 - i])
                self.Delta = self.Delta.append((t(self.Theta[self.layerNum-3-i]) @ self.Delta[-1])@ self.sig_d)
            else:
                self.dE_dTheta = self.dE_dTheta.append(self.Output[self.layerNum-2-i] @ t(self.Delta[i]))
                self.sig_d = (self.Output[self.layerNum - 3 - i]) @ (1 - self.Output[self.layerNum - 3 - i])
                self.Delta = self.Delta.append((t(self.Theta[self.layerNum-3-i]) @ self.Delta[-1])@ self.sig_d)
                self.dE_dTheta = mean(self.dE_dTheta, 2)
                
         
        
    def updateParams(self, eta):
        for i in range(self.layerNum - 1):
            self.Theta[i] = self.Theta[i] - eta * self.dE_dTheta[i]

