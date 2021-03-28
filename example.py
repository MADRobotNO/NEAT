from Neat import Neat

neat = Neat(2, 1, 10)
from RandomData import Xor
xor = Xor()
print(xor.data)
print(xor.targets)
for epoch in range(3):
    print("Epoch:", epoch)
    neat.fit_models(xor.data)
    neat.mutate_models()
