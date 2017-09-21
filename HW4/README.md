# BME 595 HW4
## Conclusion
### Neural Network Design:

I use the same parameters for both cases:
- 3 Layers: 784 -> 98 -> 10
- Activation function: Sigmoid
- Learning rate: 0.1
- Loss function: MSE


### Test error rate: 

Using nn package will be monotonic decreasing from ~9% to ~3% as epoch number goes up, whereas in my own designed case, the error rate keeps fluctuating between ~4% to ~23%. Perhaps using learning rate = 0.1 is fine for nn package, but too big for my NN.
### Operation time: 

Both cases have similar scale (100 to 1000 seconds) and show linear growth with respect to epoch number, as nn package consumes slightly more time to run.
## NeuralNetwork.py 
### NeuralNetwork class

#### __init__(self, layerSize = []):
  - layerSize: the input list of layers
  - Theta = []
  - dE_dTheta = []
  
  - Initialize the theta list with size: (layerSize[i]+1, layerSize[i+1])
  ```python
  self.theta.append(randn(layerSize[i] + 1, layerSize[i + 1]))
  ```
#### getLayer(self, layer):
  - return the theta tensor at that layer
  
#### forward(self, Input):
  - a, z = [], []
  - if-else condition is used to differentiate matrix-vector and matrix-matrix multiplication. Every layer of theta is transposed before conducting multiplication. (theta' X Input)
  - Feed each output into sigmoid function, and the output of the current layer will be the input to the next layer, and store the outputs of each layer in a and z lists.
  ```python
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
  ```
  
#### backward(self, target):
  - Use the list "a" that created in forward to help calculate theta and dE_dTheta of each layer 
  - if-else condiction is used to decide whether we have one/multiple samples. 
  ```python
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
  ```

#### updateParams(self, eta):
  - for loop to update each layer of theta with respect to the given learning rate eta
  ```python
  for i in range(self.layerNum - 1):
            self.Theta[i] = self.Theta[i] - eta * self.dE_dTheta[i]
  ```

## my_img2num.py
### MyImg2Num class
  #### train(self)
  - Load the MNIST data and set batch size to 20
  - Create NeuralNetwork object MI2N, and set size to [784, 98, 10]
  - For each epoch, conduct forward-backward-update cycle to train MI2N 
  - Set learning rate = 0.1
  ```python
  self.MI2N = NeuralNetwork([784, 98, 10])
        
        for e in range(self.epoch):
            for batch_index, (data, label) in enumerate(self.TrainLoader):
                Data = torch.zeros(784, 20)
                for batch in range(20):
                    Data[:, batch] = data[batch][0].view(784)
                
                target = torch.zeros(10, 20)
                for i in range(20):
                    target[label[i], i] = 1
                self.MI2N.forward(Data)
                self.MI2N.backward(target)
                self.MI2N.updateParams(0.1)
  ```
  
  #### forward(self, img)
  - forward function for test sets
  ```python
  Output = self.MI2N.forward(img.view(784))
        value, inx = torch.max(Output, 0)
        return inx
  ```

### Test training results
- Using custom test file
- Set batch size to 1
- Test epoch from 10 to 100
- Count the opeation time and error rate for each run

![alt text](https://github.com/chenpine/BME595_Deep_Learning/blob/master/HW4/myNN_error%20rate.png "Error Rate vs. Epoch")
![alt text](https://github.com/chenpine/BME595_Deep_Learning/blob/master/HW4/myNN_operation%20time.png "Operation Time vs. Epoch")

## nn_img2num.py
### NnImg2Num class
  #### __init__(self)
  - Using nn.Module constructor
  - Building two nn.Linear modules with size (784, 98) and (98, 10)
  ```python
  super(NnImg2Num, self).__init__()
        self.fc1 = nn.Linear(784, 98)
        self.fc2 = nn.Linear(98, 10)
  ```
  
  #### train(self)
  - Load the MNIST data and set batch size to 20
  - Use optim package to update the parameters and set learning rate = 0.1
  - Run forward-backward-update cycle with nn functions
  ```python
  optimizer = optim.SGD(self.parameters(), lr = 0.1)
        for e in range(self.epoch):
            for batch_index, (data, label) in enumerate(self.TrainLoader):
                optimizer.zero_grad()
                
                Data = torch.zeros(20, 784)
                for batch in range(20):
                    Data[batch,:] = data[batch][0].view(784)
                    
                z1 = self.fc1(Variable(Data))
                sg = nn.Sigmoid()
                h1 = sg(z1)
                output = self.fc2(h1)
                
                target = torch.zeros(20, 10)
                for i in range(20):
                    target[i, label[i]] = 1
                target = Variable(target)
                loss_o = nn.MSELoss()
                loss = loss_o(output, target)
                loss.backward()
                optimizer.step()
  ```
  #### forward(self, img)
  - forward function for test sets
  ```python
  Input = Variable(img.view(784))
        z1 = self.fc1(Input)
        sg = nn.Sigmoid()
        h1 = sg(z1)
        z2 = self.fc2(h1)
        Output = sg(z2)
        value, inx = torch.max(Output, 0)
        return inx
  ```
  ### Test Training results
  - Using custom test file
  - Set batch size to 1
  - Test epoch from 10 to 100
  - Count the opeation time and error rate for each run

![alt text](https://github.com/chenpine/BME595_Deep_Learning/blob/master/HW4/NN_error%20rate.png "Error Rate vs. Epoch")
![alt text](https://github.com/chenpine/BME595_Deep_Learning/blob/master/HW4/NN_operation%20time.png "Operation Time vs. Epoch")


  
