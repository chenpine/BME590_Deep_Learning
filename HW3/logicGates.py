from NeuralNetwork import NeuralNetwork
import torch

class AND():
    def __init__(self):
        self.And = NeuralNetwork([2, 1])
        self.theta_0 = self.And.getLayer(0)
        
    def train(self):
        for i in range(10):
            self.forward(True, True)
            self.And.backward(1)
            self.And.updateParams(0.1)
            
            self.forward(True, False)
            self.And.backward(0)
            self.And.updateParams(0.1)
            
            self.forward(False, True)
            self.And.backward(0)
            self.And.updateParams(0.1)
            
            self.forward(False, False)
            self.And.backward(0)
            self.And.updateParams(0.1)
            
            print("The value of theta of " + i + " round is: " + self.theta_0)
            i+=1
        
        
    def forward(self, *Input):
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
        self.Or = NeuralNetwork([2, 1])
        self.theta_0 = self.Or.getLayer(0)
    
    def train(self):
        for i in range(10):
            self.forward(True, True)
            self.Or.backward(1)
            self.Or.updateParams(0.1)
            
            self.forward(True, False)
            self.Or.backward(1)
            self.Or.updateParams(0.1)
            
            self.forward(False, True)
            self.Or.backward(1)
            self.Or.updateParams(0.1)
            
            self.forward(False, False)
            self.Or.backward(0)
            self.Or.updateParams(0.1)
            
            print("The value of theta of " + i + " round is: " + self.theta_0)
            i+=1
        
    def forward(self, *Input):
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
        self.Not = NeuralNetwork([1, 1])
        self.theta_0 = self.Not.getLayer(0)
    
    def train(self):
        for i in range(10):
            self.forward(True)
            self.Not.backward(0)
            self.Not.updateParams(0.1)
            
            self.forward(False)
            self.Not.backward(1)
            self.Not.updateParams(0.1)
            
            print("The value of theta of " + i + " round is: " + self.theta_0)
            i+=1
        
        
    def forward(self, *Input):
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
        self.Xor = NeuralNetwork([2, 2, 1])
        self.theta_0 = self.Xor.getLayer(0)
        self.theta_1 = self.Xor.getLayer(1)
        
    def train(self):
        for i in range(10):
            self.forward(True, True)
            self.Xor.backward(False)
            self.Xor.updateParams(0.1)
            
            self.forward(True, False)
            self.Xor.backward(True)
            self.Xor.updateParams(0.1)
            
            self.forward(False, True)
            self.Xor.backward(True)
            self.Xor.updateParams(0.1)
            
            self.forward(False, False)
            self.Xor.backward(False)
            self.Xor.updateParams(0.1)
            
            print("The value of theta of " + i + " round is: " + self.theta_0)
            i+=1
        
        
    def forward(self, *Input):
        Tensor_Input = torch.zeros(len(Input))
        for i in range(len(Input)):
            Tensor_Input[i] = 1 if Input[i] == True else 0
        self.Output = self.Xor.forward(Tensor_Input)
        if self.Output[0] < 0.5:
            return False
        else:
            return True
        
# Instantiate the objects        
And = AND()
Or = OR()
Not = NOT()
Xor = XOR()

# Train the neural networks
And.train()
Or.train()
Not.train()
Xor.train()

# Test training results for each gates
print("And(True, True): ", And(True, True))
print("And(False, True): ", And(False, True))
print("And(True, False): ", And(True, False))
print("And(False, False): ", And(False, False))

print("Not(True): ", Not(True))
print("Not(False): ", Not(False))

print("Or(True, True): ", Or(True, True))
print("Or(False, True): ", Or(False, True))
print("Or(True, False): ", Or(True, False))
print("Or(False, False): ", Or(False, False))

print("Xor(True, True): ", Xor(True, True))
print("Xor(False, True): ", Xor(False, True))
print("Xor(True, False): ", Xor(True, False))
print("Xor(False, False): ", Xor(False, False))