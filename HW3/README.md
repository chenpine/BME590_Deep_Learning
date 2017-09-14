# BME 595 HW3
## NeuralNetwork.py (NeuralNetwork Class)

### __init__(self, layerSize = []):
  - layerSize: the input list of layers
  - Theta = []
  - dE_dTheta = []
  
  - Initialize the theta dictionary with size: (layerSize[i]+1, layerSize[i+1])
  ```python
  self.theta.append(randn(layerSize[i] + 1, layerSize[i + 1]))
  ```
### getLayer(self, layer):
  - return the theta tensor at that layer
  
### forward(self, Input):
  - Output = []
  - if-else condition is used to differentiate matrix-vector and matrix-matrix multiplication. Every layer of theta is transposed before conducting multiplication. (theta' X Input)
  - Feed each output into sigmoid function, and the output of the current layer will be the input to the next layer
  ```python
  self.Input = Input
  self.Output = []
  for i in range(self.layerNum - 1):
            if (len(self.Input.shape) == 1):
                self.Output = self.Output.append(1 / (1 + exp(0 - mv(t(self.Theta[i]), cat((ones(1), self.Input))))))
                self.Input = self.Output[-1]
            else:
                self.Output = self.Output.append(1 / (1 + exp(0 - mm(t(self.Theta[i]), cat((ones(1, self.Input.shape[1]), self.Input))))))
                self.Input = self.Output[-1]
  ```
  
### backward(self, target):
  - calculate the delta of the last layer (Delta_last) first then run foor loop to backpropagate to get Delta for other Delta and dE_dTheta
  - if-else condiction is used to decide whether we have one/multiple samples. In the multiple samples condition, we need to take average of dE_dTheta
  ```python
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
                

  ```

### updateParams(self, eta):
  - for loop to update each layer of theta with respect to the given learning rate eta
  ```python
  for i in range(self.layerNum - 1):
            self.Theta[i] = self.Theta[i] - eta * self.dE_dTheta[i]
  ```

## logicGates.py
  - __init__(self): Use NeuralNetwork class to create object for each logic gate class and initialize theta using getLayer method.
  - train(self): Use forward-backward-updateParams cycle in for-loop to train the logic gates.
  - forward(self, *Input): Convert the boolean input to numbers and feed to NeuralNetwork version of forward. Return boolean output.

### AND class
  - in = 2
  - out = 1
  ```python
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
  ```
### OR class
  - in = 2
  - out = 1
  ```python
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
  ```

### NOT class
  - in = 1
  - out = 1
  ```python
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
  ```

### XOR class
  - in = 2
  - **h1 = 2**
  - out = 1
  ```python
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

  ```
### Test training results
  ```python
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
