import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import cv2



class img2obj(nn.Module):
    #def __init__(self):
        
        
    def train(self):
        # LeNet-5 building
        super(img2obj, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 100)
        
        # Load CIFAR-100 training data
        self.TrainLoader = torch.utils.data.DataLoader(datasets.CIFAR100('../data', train = True, download = True, 
                                                          transform = transforms.Compose([
                                                                  transforms.ToTensor(), 
                                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                                  ])), 
                                           batch_size = 100, shuffle = True)
        
        # Copy from nn_img2num.py
        optimizer = optim.SGD(self.parameters(), lr = 0.7)
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
                
                target = torch.zeros(100, 100)
                for i in range(100):
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

   
    def cam(self):    
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("test")
        self.img_counter = 0
        while True:
            ret, frame = cam.read()
            cv2.imshow("test", frame)
            if not ret:
                break
            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "opencv_frame_{}.png".format(self.img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                self.img_counter += 1        
        cam.release()
        cv2.destroyAllWindows()

    def view(self, img):
        CIFAR_label = ["beaver", "dolphin", "otter", "seal", "whale", "aquarium fish", "flatfish", 
               "ray", "shark", "trout", "orchids", "poppies", "roses", "sunflowers", "tulips", 
               "bottles", "bowls", "cans", "cups", "plates", 'apples', 'mushrooms', 'oranges', 'pears', 
               "sweet peppers", "clock", "computer keyboard", "lamp", "telephone", 'television', 'bed', 'chair', 
               'couch', 'table', 'wardrobe', 'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 'bear', 'leopard',
               'lion', 'tiger', 'wolf', 'bridge', 'castle', 'house', 'road', 'skyscraper', 'cloud', 'forest', 'mountain', 
               'plain', 'sea', 'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', 'fox', 'porcupine', 'possum', 
               'raccoon', 'skunk', 'crab', 'lobster', 'snail', 'spider', 'worm', 'baby', 'boy', 'girl', 'man', 'woman',
               'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
               'maple', 'oak', 'palm', 'pine', 'willow', 'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', 'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']
        Img = Image.open(img)
        Tensor = transforms.ToTensor()
        Input = Tensor(Img)
        Input.resize_(1, 3, 32, 32)
        Output = F.max_pool2d(self.conv1(Variable(Input)), 2)
        Output = F.relu(Output)
        Output = F.max_pool2d(self.conv2(Output), 2)
        Output = F.relu(Output)
        Output = Output.view(Output.size(0), -1)
        Output = F.relu(self.fc1(Output))
        Output = F.relu(self.fc2(Output))
        Output = self.fc3(Output)
        value, index = torch.max(Output, 1)
        index = index.data.int().numpy().sum()
        draw = ImageDraw.Draw(Img)
        font = ImageFont.truetype("Arial.ttf", 30)
        draw.text((0, 0), CIFAR_label[index], (0, 0, 0), font = font)
        Img.save(img)

# Create model object and train the CNN to produce caption
model = img2obj()
model.epoch = 30
model.train() 
model.cam()
for i in range(model.img_counter):
    model.view('opencv_frame_{}.png'.format(i))