from neural_network import NeuralNetwork
import torch

class AND():
    def __init__(self):
        #super().__init__(layerSize = [3, 1])
        #NeuralNetwork.__init__(self, layerSize = [3, 1])
        #self.theta[0] = self.getLayer(0)
        #self.theta[0] = torch.DoubleTensor([-20],[15],[15])
        #NeuralNetwork.theta[0] = self.getLayer(0)
        #NeuralNetwork.theta[0] = torch.DoubleTensor([-20],[15],[15])
        self.And = NeuralNetwork([2, 1])
        self.theta_0 = self.And.getLayer(0)
        self.theta_0[0, 0] = -20
        self.theta_0[1, 0] = 15
        self.theta_0[2, 0] = 15
        #self.theta_0 = torch.DoubleTensor([[-20], [15], [15]])
        
    def __call__(self, *arg):
        return self.forward(arg)
        
    def forward(self, Input):
        Tensor_Input = torch.zeros(len(Input))
        for i in range(len(Input)):
            Tensor_Input[i] = 1 if Input[i] == True else 0
        self.Output = self.And.forward(Tensor_Input)
        if self.Output[0] < 0.5:
            return False
        else:
            return True
        
class OR():
    def __init__(self):
        #super().__init__(self, layerSize = [3, 1])
        #NeuralNetwork.__init__(self, layerSize = [3, 1])
        #self.theta[0] = self.getLayer(0)
        #self.theta[0] = torch.DoubleTensor([[-10],[15],[15]])
        self.Or = NeuralNetwork([2, 1])
        self.theta_0 = self.Or.getLayer(0)
        #self.theta_0 = torch.DoubleTensor([[-10], [15], [15]])
        self.theta_0[0, 0] = -10
        self.theta_0[1, 0] = 15
        self.theta_0[2, 0] = 15
        
    def __call__(self, *arg):
        return self.forward(arg)
        
    def forward(self, Input):
        Tensor_Input = torch.zeros(len(Input))
        for i in range(len(Input)):
            Tensor_Input[i] = 1 if Input[i] == True else 0
        self.Output = self.Or.forward(Tensor_Input)
        if self.Output[0] < 0.5:
            return False
        else:
            return True
        
class NOT():
    def __init__(self):
        #super().__init__(self, layerSize = [2, 1])
        #NeuralNetwork.__init__(self, layerSize = [2, 1])
        #self.theta[0] = self.getLayer(0)
        #self.theta[0] = torch.DoubleTensor([[10],[-15]])
        self.Not = NeuralNetwork([1, 1])
        self.theta_0 = self.Not.getLayer(0)
        #self.theta_0 = torch.DoubleTensor([[10], [-15]])
        self.theta_0[0, 0] = 10
        self.theta_0[1, 0] = -15
        
    def __call__(self, *arg):
        return self.forward(arg)
        
    def forward(self, Input):
        Tensor_Input = torch.zeros(len(Input))
        for i in range(len(Input)):
            Tensor_Input[i] = 1 if Input[i] == True else 0
        self.Output = self.Not.forward(Tensor_Input)
        if self.Output[0] < 0.5:
            return False
        else:
            return True
        
class XOR():
    def __init__(self):
        #super().__init__(self, layerSize = [3, 3, 1])
        #NeuralNetwork.__init__(self, layerSize = [3, 3, 1])
        #self.theta[0] = self.getLayer(0)
        #self.theta[0] = torch.DoubleTensor([[-10, 15],[20, -10],[20, -10]])
        #self.theta[1] = self.getLayer(1)
        #self.theta[1] = torch.DoubleTensor([[-20],[15],[15]])
        self.Xor = NeuralNetwork([2, 2, 1])
        self.theta_0 = self.Xor.getLayer(0)
        self.theta_0[0, 0] = -10
        self.theta_0[1, 0] = 20
        self.theta_0[2, 0] = 20
        self.theta_0[0, 1] = 15
        self.theta_0[1, 1] = -10
        self.theta_0[2, 1] = -10
        #self.theta_0 = torch.DoubleTensor([[-10, 15], [20, -10], [20, -10]])
        self.theta_1 = self.Xor.getLayer(1)
        self.theta_1[0, 0] = -20
        self.theta_1[1, 0] = 15
        self.theta_1[2, 0] = 15
        #self.theta_1 = torch.DoubleTensor([[-20], [15], [15]])
        
    def __call__(self, *arg):
        return self.forward(arg)
        
    def forward(self, Input):
        Tensor_Input = torch.zeros(len(Input))
        for i in range(len(Input)):
            Tensor_Input[i] = 1 if Input[i] == True else 0
        self.Output = self.Xor.forward(Tensor_Input)
        if self.Output[0] < 0.5:
            return False
        else:
            return True
