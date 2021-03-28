import numpy as np


class Connection:

    ENABLED = 1
    DISABLED = 0

    def __init__(self, from_node, to_node, innovation_id):
        self.weight = np.random.uniform(-1, 1)
        self.from_node = from_node
        self.to_node = to_node
        self.innovation_id = innovation_id
        self.status = self.ENABLED

    def __str__(self):
        return "From: " + str(self.from_node) + ", to: " + str(self.to_node) + ", innovation ID: " + \
               str(self.innovation_id) + ", weight: " + str(self.weight) + ", status: " + str(self.status)
