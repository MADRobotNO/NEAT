from Neat import Neat

neat = Neat(2, 1, 100, 0.3)
from RandomData import Xor
xor = Xor()

neat.fit_models(xor.data, xor.targets, number_of_epochs=100)
