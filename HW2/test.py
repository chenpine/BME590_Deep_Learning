from neural_network import NeuralNetwork
from logic_gates import AND, NOT, OR, XOR

And = AND()
Not = NOT()
Or = OR()
Xor = XOR()

print("And(True, True): ", And(True, True))
print("And(False, True): ", And(False, True))
print("And(True, False): ", And(True, False))
print("And(False, False): ", And(False, False))

print("Not(True): ", Not(True))
print("Not(False): ", Not(False))

print("Or(True, True): ", Or(True, True))
print("Or(False, True): ", Or(False, True))
print("Or(True, False): ", Or(True, False))
print("Or(False, False): ", Or(False, False))

print("Xor(True, True): ", Xor(True, True))
print("Xor(False, True): ", Xor(False, True))
print("Xor(True, False): ", Xor(True, False))
print("Xor(False, False): ", Xor(False, False))