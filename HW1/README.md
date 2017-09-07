# BME 595 HW1


## Part A
 
### Task 1:   
- in_channel: 3
- o_channel: 1 
- kernel_size: 3
- stride: 1
- mode: "known"
- kernel to use: K1
#### 1280 X 720: 
**Number of Operations:**
917,604 (#)


#### 1920 X 1080:
**Number of Operations:**
2,067,604 (#)


### Task 2:   
- in_channel: 3
- o_channel: 2 
- kernel_size: 5
- stride: 1
- mode: "known"
- kernel to use: K4, K5
#### 1280 X 720:
**Number of Operations:**
1,827,232 (#)


#### 1920 X 1080: 
**Number of Operations:**
4,123,232 (#)


### Task 3:   
- in_channel: 3
- o_channel: 3 
- kernel_size: 3
- stride: 2
- mode: "known"
- kernel to use: K1, K2, K3   
#### 1280 X 720:
**Number of Operations:**
688,203 (#)


#### 1920 X 1080: 
**Number of Operations:**
1,550,703 (#)



## Part B 

- in_channel: 3
- o_channel: 2^i, i = 0, 1, ...10 
- kernel_size: 3
- stride: 1
- mode: "rand"
- kernel to use: K1

### i = 0:

#### 1280 X 720:
**Operation Time:**
~ 250 (sec)

#### 1920 X 1080:
**Operation Time:**
~ 450 (sec)


### i = 1:

#### 1280 X 720:
**Operation Time:**
~ 430 (sec)

#### 1920 X 1080:
**Operation Time:**
~ 950 (sec)


### i = 2:

#### 1280 X 720:
**Operation Time:**
~ 800 (sec)

#### 1920 X 1080:
**Operation Time:**
~ 1950 (sec)

### (i = 3 ~ 10) will take too long and the trend is clear to be seen (exponential growth)


## Part C 

- in_channel: 3
- o_channel: 2 
- kernel_size: 3, 5, ...11
- stride: 1
- mode: "rand"
- kernel to use: K4, K5

### kernel_size = 3:

#### 1280 X 720:
**Operation Time:**
~ 300 (sec)

#### 1920 X 1080:
**Operation Time:**
~ 1000 (sec)


### kernel_size = 5:

#### 1280 X 720:
**Operation Time:**
~ 1000 (sec)

#### 1920 X 1080:
**Operation Time:**
~ 2500 (sec)


### kernel_size = 7:

#### 1280 X 720:
**Operation Time:**
~ 2000 (sec)

#### 1920 X 1080:
**Operation Time:**
~ 6000 (sec)

### (kernel_size = 9, 11) will take too long and the trend is clear to be seen (exponential growth)
