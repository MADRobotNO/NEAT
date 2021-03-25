from Perceptron import Node
from Connection import Connection
import numpy as np


class Model:

    nodes = []
    input_nodes = []
    hidden_nodes = []
    output_nodes = []
    connections = []

    def __init__(self, number_of_inputs, numeber_of_outputs, initial_mutation=False, inital_mutation_level=0.1):
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = numeber_of_outputs
        self.current_number_of_nodes = self.number_of_inputs + self.number_of_outputs
        self.initialize_nodes()
        self.initialize_connections(initial_mutation, inital_mutation_level)

    def initialize_nodes(self):
        for i in range(self.number_of_inputs):
            node = Node(len(self.nodes))
            self.nodes.append(node)
            self.input_nodes.append(node)
        for i in range(self.number_of_outputs):
            node = Node(len(self.nodes))
            self.nodes.append(node)
            self.output_nodes.append(node)

    def initialize_connections(self, initial_mutation, inital_mutation_level):
        if initial_mutation:
            for input_node in self.input_nodes:
                for output_node in self.output_nodes:
                    if np.random.rand() > inital_mutation_level:
                        self.connections.append(Connection(input_node.id, output_node.id))
                    else:
                        pass
        else:
            for input_node in self.input_nodes:
                for output_node in self.output_nodes:
                    self.connections.append(Connection(input_node.id, output_node.id))

    def mutate(self):
        pass

