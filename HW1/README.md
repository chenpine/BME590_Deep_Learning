# BME 590 HW1


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



## Part B (set o_channel to 2^i, i = 0, 1, ...10 and mode = 'rand')
