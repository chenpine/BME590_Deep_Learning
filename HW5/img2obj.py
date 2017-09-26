import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Variable

class img2obj():
    def train(self):
        # Load CIFAR100 training data
        self.TrainLoader = torch.utils.data.DataLoader(datasets.CIFAR100('../data', train = True, download = True, 
                                                          transform = transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                                  ])), 
                                           batch_size = 20, shuffle = True)
    
        # LeNet-5 building
        super(img2obj, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 100)
        
        loss_o = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr = 0.01)
        
        for epoch in range(10):
            for batch_index, (data, label) in enumerate(self.TrainLoader):
                data, label = Variable(data), Variable(label)
                optimizer.zero_grad()
                output = self.forward(data)
                loss = loss_o(output, label)
                loss.backward()
                optimizer.step()
                
    
    def forward(self, img):
        img = F.max_pool2d(F.relu(self.conv1(img)))
        img = F.max_pool2d(F.relu(self.conv2(img)))
        img = img.view(-1, 400)
        img = F.relu(self.fc1(img))
        img = F.relu(self.fc2(img))
        img = self.fc3(img)

TestLoader = torch.utils.data.DataLoader(datasets.CIFAR100('../data', train = False, download = False, 
                                                          transform = transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                                  ])), 
                                           batch_size = 10, shuffle = True)
model = img2obj()
correct, total = 0, 0
for images, labels in TestLoader:
    output = model.train(Variable(images))
    value, predict = torch.max(output.data, 1)
    total += labels.size(0)
    correct += (predict == labels).sum()
    
print('Accuracy of the network on the test sets is: ', 100* correct/total, "%")