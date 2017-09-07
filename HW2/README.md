# BME 595 HW2
## neural_network.py (NeuralNetwork Class)
### __init__(self, layerSize = []):
  - layerSize: the input list of layers
  - Initialize the theta dictionary with size: (layerSize[i]+1, layerSize[i+1])
  ```python
  self.theta.append(randn(layerSize[i] + 1, layerSize[i + 1]))
  ```
### forward(self, Input):
  - if-else condition is used to differentiate matrix-vector and matrix-matrix multiplication. Every layer of theta is transposed before conducting multiplication. (theta' X Input)
  - Feed each output into sigmoid function.
  ```python
  self.Output = 1 / (1 + exp(0 - mm(t(self.theta[i]), cat((ones(1, self.Output.shape[1]), self.Output)))))
  ```
  
## logic_gates.py 
### AND class
- in = 2
- out = 1
- theta[0] = [[-20], [15], [15]]
```python
self.And = NeuralNetwork([2, 1])
        self.theta_0 = self.And.getLayer(0)
        self.theta_0[0, 0] = -20
        self.theta_0[1, 0] = 15
        self.theta_0[2, 0] = 15
```

### OR class
- in = 2
- out = 1
- theta[0] = [[-10], [15], [15]]
```python
self.Or = NeuralNetwork([2, 1])
        self.theta_0 = self.Or.getLayer(0)
        self.theta_0[0, 0] = -10
        self.theta_0[1, 0] = 15
        self.theta_0[2, 0] = 15
```

### NOT class
- in = 1
- out = 1
- theta[0] = [[10], [-15]]
```python
self.Not = NeuralNetwork([1, 1])
        self.theta_0 = self.Not.getLayer(0)
        self.theta_0[0, 0] = 10
        self.theta_0[1, 0] = -15
```


### XOR class
- in = 2
- **h1 = 2**
- out = 1
- theta[0] = [[-10, 15], [20, -10], [20, -10]]
- theta[1] = [[-20], [15], [15]]
```python
self.And = NeuralNetwork([2, 1])
self.Xor = NeuralNetwork([2, 2, 1])
        self.theta_0 = self.Xor.getLayer(0)
        self.theta_0[0, 0] = -10
        self.theta_0[1, 0] = 20
        self.theta_0[2, 0] = 20
        self.theta_0[0, 1] = 15
        self.theta_0[1, 1] = -10
        self.theta_0[2, 1] = -10
        
        self.theta_1 = self.Xor.getLayer(1)
        self.theta_1[0, 0] = -20
        self.theta_1[1, 0] = 15
        self.theta_1[2, 0] = 15
```

## test.py
- Instantiate each logic and test all the combinations
