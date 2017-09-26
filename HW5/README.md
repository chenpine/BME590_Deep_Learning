# BME 595 HW5

## img2num.py
### img2num class
#### train(self)
- Use nn.Module 
- Load the MNIST data and set batch size to 20
- 2 Convolution layers + 2 Pooling layers + 3 linear layers + 4 relu activations
- Use optim package to update the parameters and set learning rate = 0.01
- Run forward-backward-update cycle with nn method
```python
self.conv1 = nn.Conv2d(1, 6, 5)
self.conv2 = nn.Conv2d(6, 16, 5)
self.fc1 = nn.Linear(256, 120)
self.fc2 = nn.Linear(120, 84)
self.fc3 = nn.Linear(84, 10)

optimizer = optim.SGD(self.parameters(), lr = 0.01)
        for e in range(self.epoch):
            for batch_index, (data, label) in enumerate(self.TrainLoader):
                optimizer.zero_grad()
                
                #Data = torch.zeros(20, 784)
                #for batch in range(20):
                    #Data[batch,:] = data[batch][0].view(784)
                     
                output = F.max_pool2d(self.conv1(Variable(data)), 2)
                output = F.relu(output)
                output = F.max_pool2d(self.conv2(output), 2)
                output = F.relu(output)
                output = output.view(output.size(0), -1)
                output = F.relu(self.fc1(output))
                output = F.relu(self.fc2(output))
                output = self.fc3(output)
                
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
Output = F.max_pool2d(self.conv1(Variable(img)), 2)
Output = F.relu(Output)
Output = F.max_pool2d(self.conv2(Output), 2)
Output = F.relu(Output)
Output = Output.view(Output.size(0), -1)
Output = F.relu(self.fc1(Output))
Output = F.relu(self.fc2(Output))
Output = self.fc3(Output)
value, inx = torch.max(Output, 1)
return inx


## img2obj.py
### img2obj class
#### train(self)
- 
