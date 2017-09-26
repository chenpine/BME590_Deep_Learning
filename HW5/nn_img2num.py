import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable





class NnImg2Num(nn.Module):
    def __init__(self):    
        super(NnImg2Num, self).__init__()
        self.fc1 = nn.Linear(784, 98)
        self.fc2 = nn.Linear(98, 10)
        
    def train(self):
        self.TrainLoader = torch.utils.data.DataLoader(datasets.MNIST('../data', train = True, download = True, 
                                                          transform = transforms.Compose([
                                                                  transforms.ToTensor(), 
                                                                  #transforms.Normalize((0.1307,), (0.3801,))
                                                                  ])), 
                                           batch_size = 20, shuffle = True)
        
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
        
        
    def forward(self, img):
        Input = Variable(img.view(784))
        z1 = self.fc1(Input)
        sg = nn.Sigmoid()
        h1 = sg(z1)
        z2 = self.fc2(h1)
        Output = sg(z2)
        value, inx = torch.max(Output, 0)
        return inx




    