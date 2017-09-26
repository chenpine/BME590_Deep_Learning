import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Variable

class img2num(nn.Module):
    def __init__(self):
        
        # LeNet-5 building
        super(img2num, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def train(self):
        # Load MNIST training data
        self.TrainLoader = torch.utils.data.DataLoader(datasets.MNIST('../data', train = True, download = True, 
                                                          transform = transforms.Compose([
                                                                  transforms.ToTensor(), 
                                                                  transforms.Normalize((0.1307,), (0.3801,))
                                                                  ])), 
                                           batch_size = 20, shuffle = True)
        
        # Copy from nn_img2num.py
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
        
    def forward(self, img):
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