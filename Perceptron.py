import numpy as np


class Node:

    def __init__(self, id):
        self.id = id
        self.bias = np.random.rand()

    def __str__(self):
        return "Node id: " + str(self.id) + ",bias: " + str(self.bias)
