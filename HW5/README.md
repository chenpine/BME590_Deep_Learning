# BME 595 HW5

## Conclusion
In MNIST case, CNN has higher accuracy but take longer time to train.
![alt text](https://github.com/chenpine/BME595_Deep_Learning/blob/master/HW4/NN_error%20rate.png "NN: Error Rate vs. Epoch")
![alt text](https://github.com/chenpine/BME595_Deep_Learning/blob/master/HW4/NN_operation%20time.png "NN: Operation Time vs. Epoch")
![alt text](https://github.com/chenpine/BME595_Deep_Learning/blob/master/HW5/ErrorRate_LeNet_MNIST.png "CNN: Error Rate vs. Epoch")
![alt text](https://github.com/chenpine/BME595_Deep_Learning/blob/master/HW5/OperationTime_LeNet_MNIST.png "CNN: Operation Time vs. Epoch")


In CIFAR-100 training case, the test accuracy is low even using lr = 0.7 and up to 60 epoch.
![alt text](https://github.com/chenpine/BME595_Deep_Learning/blob/master/HW5/CIFAR_lr%3D0.7_Error.png "CNN: Error Rate vs. Epoch")
![alt text](https://github.com/chenpine/BME595_Deep_Learning/blob/master/HW5/CIFAR_lr%3D0.7_Time.png "CNN: Operation Time vs. Epoch")


Nonetheless, we can still use this trained model to try to give the correct caption to the webcam-fetched image.
![alt text](https://github.com/chenpine/BME595_Deep_Learning/blob/master/HW5/opencv_frame_0.png "CNN: Error Rate vs. Epoch")
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
```

## img2obj.py
### img2obj class
#### train(self)
- Use nn.Module 
- Load the CIFAR-100 data and set batch size to 100
- 2 Convolution layers + 2 Pooling layers + 3 linear layers + 4 relu activations
- Use optim package to update the parameters and set learning rate = 0.7, which is relatively high
- Run forward-backward-update cycle with nn method
```python
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
```


#### cam(self)
- Use cv2 package 
- Capture image when SPACE key is pressed
- Stop the camera when ESC key is pressed
- Save the image in current folder
```python
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
```


#### view(self, img)
- Manually list the labels
- Load the image from the folder
- Feed into the CNN and produce caption
- Save the image
```python
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
```

#### Test case
```python
model = img2obj()
model.epoch = 30
model.train() 
model.cam()
for i in range(model.img_counter):
    model.view('opencv_frame_{}.png'.format(i))
```


