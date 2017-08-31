from conv import Conv2D
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import time
import math
import matplotlib.pyplot as plt

s_img = "1280_720"
l_img = "1920_1080"
#t_img = "100_100"
imgSet = [s_img, l_img]
#imgSet = [t_img]
for img in imgSet:
    PIL_img = Image.open(img + ".jpg")
    ToTensor = transforms.ToTensor()
    ToImage = transforms.ToPILImage()
    input_FT = ToTensor(PIL_img)

# Part A

# Task 1
    convA1 = Conv2D(3, 1, 3, 1, "known")
    no_opsA1, o_FTA1 = convA1.forward(input_FT)
    o_imgA1 = ToImage(o_FTA1)
    o_imgA1.save(img + "PartA_Task1.jpg")
    print(img, "Number of Operations of Task 1: ", no_opsA1)

# Task 2
    convA2 = Conv2D(3, 2, 5, 1, "known")
    no_opsA2, o_FTA2 = convA2.forward(input_FT)
    for chan in range(2):
        o_imgA2 = ToImage(o_FTA2[chan].unsqueeze(0))
        o_imgA2.save(img + "Channel " + str(chan) + "_PartA_Task2.jpg")
    print(img, "Number of Operations of Task 2: ", no_opsA2)
    
# Task 3
    convA3 = Conv2D(3, 3, 3, 2, "known")
    no_opsA3, o_FTA3 = convA3.forward(input_FT)
    for i in range(3):
        o_imgA3 = ToImage(o_FTA3[i].unsqueeze(0))
        o_imgA3.save(img + "Channel " + str(i) + "_PartA_Task3.jpg")
    print(img, "Number of Operations of Task 3: ", no_opsA3)

# Part B
    no_fig = 1
    total_t, no_ch = [], []
    for i in range(3):
        print ("PartB", img, i)
        start = time.time();
        convB = Conv2D(3, (int)(math.pow(2, i)), 3, 1, "rand")
        no_opsB, o_FTB = convB.forward(input_FT)
        duration = time.time() - start
        total_t.append(duration)
        no_ch.append((int)(math.pow(2, i)))
    
    plt.figure(no_fig)
    plt.xlabel('Number of Channels')
    plt.ylabel('Total Operation Time')
    plt.title("Operation Time of Different Channel Numbers " + img)
    plt.plot(no_ch, total_t)
    no_fig += 1
    
#Part C
    total_t, no_ch = [], []
    for i in range(3, 9, 2):
        print ("PartC", img, i)
        start = time.time();
        convC = Conv2D(3, 2, i, 1, "rand")
        no_opsC, o_FTC = convC.forward(input_FT)
        duration = time.time() - start
        total_t.append(duration)
        no_ch.append(i)
    
    plt.figure(no_fig)
    plt.xlabel('Kernel Size')
    plt.ylabel('Total Operation Time')
    plt.title("Operation Time of Different Kernel Size" + img)
    plt.plot(no_ch, total_t)
    no_fig += 1
plt.show()



