import numpy as np


class Connection:
    def __init__(self, from_neuron, to_neuron):
        self.weight = np.random.rand()
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.status = 1

    def __str__(self):
        return "From: " + str(self.from_neuron) + ", to: " + str(self.to_neuron) + ", weight: " + str(self.weight) + \
               ", status: " + str(self.status)
