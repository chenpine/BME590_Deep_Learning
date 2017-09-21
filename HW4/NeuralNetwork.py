from torch import t, randn, mv, mm, cat, ones, sigmoid, mul

class NeuralNetwork():
    def __init__(self, layerSize = []):
        self.Theta = []
        self.dE_dTheta = []
        self.layerSize = layerSize
        self.layerNum = len(self.layerSize)-1
        
        for i in range(self.layerNum):
            self.Theta.append(randn(layerSize[i] + 1, layerSize[i + 1]))
            
            
    def getLayer(self, layer):
        return self.Theta[layer]
        
    
    def forward(self, Input):
        self.a, self.z = [], []
        self.a.append(Input)
        self.Output = Input
        
        for i in range(self.layerNum):
            if (len(self.Output.shape) == 1):
                self.Output = cat((ones(1), self.Output))
                self.Output = mv(t(self.Theta[i]), self.Output)
                self.z.append(self.Output)
                self.Output = sigmoid(self.Output)
                self.a.append(self.Output)
            else:
                self.Output = cat((ones(1, self.Output.shape[1]), self.Output), 0)
                self.Output = mm(t(self.Theta[i]), self.Output)
                self.z.append(self.Output)
                self.Output = sigmoid(self.Output)
                self.a.append(self.Output)
                
        return self.Output
    
    def backward(self, target):
        self.target = target
        self.dE_dTheta = []
        
        for i in range(self.layerNum + 1, 0, -1):
            if (len(self.a[i-1].shape) == 1):
                self.aH = cat((ones(1), self.a[i-1]))
            else:
                self.aH = cat((ones(1, self.a[i-1].shape[1]), self.a[i-1]), 0)
            if i == self.layerNum + 1:
                self.delta = mul((self.a[-1] - self.target), mul(self.a[-1], (1 - self.a[-1])))
            elif i == self.layerNum:
                if (len(self.aH.shape) == 1):
                    self.dE_dTheta.insert(0, mm(self.aH.unsqueeze(1), self.delta.unsqueeze(0)))
                    self.delta = mul(mv(self.Theta[i-1], self.delta), mul(self.aH, 1 - self.aH))
                else:
                    self.dE_dTheta.insert(0, mm(self.aH, t(self.delta)))
                    self.delta = mul(mm(self.Theta[i-1], self.delta), mul(self.aH, 1 - self.aH))
            else: 
                if(len(self.aH.shape) == 1):
                    self.dE_dTheta.insert(0, mm(self.aH.unsqueeze(1), self.delta[1:].unsqueeze(0)))
                    self.delta = mul(mv(self.Theta[i-1], self.delta[1:]), mul(self.aH, 1 - self.aH))
                else:
                    self.dE_dTheta.insert(0, mm(self.aH, t(self.delta[1:])))
                    self.delta = mul(mm(self.Theta[i-1], self.delta[1:]), mul(self.aH, 1 - self.aH))
        
    def updateParams(self, eta):
        for i in range(self.layerNum):
            self.Theta[i] = self.Theta[i] - eta * self.dE_dTheta[i]

