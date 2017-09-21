import torch
from torchvision import datasets, transforms
from NeuralNetwork import NeuralNetwork

class MyImg2Num():
        
    def train(self):
        self.TrainLoader = torch.utils.data.DataLoader(datasets.MNIST('../data', train = True, download = True, 
                                                          transform = transforms.Compose([
                                                                  transforms.ToTensor(), 
                                                                  #transforms.Normalize((0.1307,), (0.3801,))
                                                                  ])), 
                                           batch_size = 20, shuffle = True)
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
                
    def forward(self, img):
        Output = self.MI2N.forward(img.view(784))
        value, inx = torch.max(Output, 0)
        return inx