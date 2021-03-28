import numpy as np


class Node:

    TYPE_INPUT = "input"
    TYPE_OUTPUT = "output"
    TYPE_HIDDEN = "hidden"

    def __init__(self, id, node_type):
        self.id = id
        self.node_type = node_type
        self.input = 0.0
        self.output = None

        if node_type != self.TYPE_INPUT:
            self.bias = np.random.uniform(-1, 1)
        else:
            self.bias = 0.0

    def calculate_output(self, connections, nodes):
        weights = self.get_weights(connections)
        input_data = self.get_input_data(connections, nodes)
        output_data = 0.0
        for in_data in input_data:
            for weight in weights:
                output_data += in_data * weight
        output_data += self.bias
        output_data = self.activation_function(output_data)
        self.output = output_data
        return output_data

    def activation_function(self, output_data):
        return np.tanh(output_data)

    def get_input_data(self, connections, nodes):
        input_data = []
        for connection in connections:
            for node in nodes:
                if connection.from_node == node.id:
                    input_data.append(node.output)
        return input_data

    def get_weights(self, connections):
        weights = []
        for connection in connections:
            weights.append(connection.weight)
        return weights

    def __str__(self):
        return "Node id: " + str(self.id) + ", node type: " + self.node_type + ", input: " + str(self.input) \
               + " ,bias: " + str(self.bias) + ", output: " + str(self.output)
