from Perceptron import Node
from Connection import Connection
import numpy as np


class Model:

    def __init__(self, number_of_inputs, numeber_of_outputs, model_id):
        self.model_id = model_id
        self.outputs = []
        self.innovations_array = []
        self.nodes = []
        self.input_nodes = []
        self.hidden_nodes = []
        self.output_nodes = []
        self.connections = []

        self.fitnes = 0.0

        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = numeber_of_outputs
        self.current_number_of_nodes = self.number_of_inputs + self.number_of_outputs

        self.initialize_nodes()
        self.initialize_connections()

    def initialize_nodes(self):
        for i in range(self.number_of_inputs):
            node = Node(len(self.nodes), Node.TYPE_INPUT)
            self.nodes.append(node)
            self.input_nodes.append(node)
        for i in range(self.number_of_outputs):
            node = Node(len(self.nodes), Node.TYPE_OUTPUT)
            self.nodes.append(node)
            self.output_nodes.append(node)

    def initialize_connections(self):
        for input_node in self.input_nodes:
            for output_node in self.output_nodes:
                self.create_connection([input_node, output_node])

    def mutate(self):
        select_mutation = np.random.randint(0, 5)
        if select_mutation < 1:
            self.create_connection()
        elif 1 <= select_mutation < 2:
            self.create_node()
        elif 2 <= select_mutation < 3:
            self.enable_disable_connection()
        elif 3 <= select_mutation < 4:
            self.adjust_weight()
        else:
            self.change_weight()

    def create_connection(self, input_output_nodes=None):
        innovation = None
        if input_output_nodes is None:
            random_nodes = self.select_random_nodes()
            input_node = random_nodes[0]
            output_node = random_nodes[1]
            if [input_node.id, output_node.id] in self.innovations_array or [output_node.id, input_node.id] in self.innovations_array:
                pass
            else:
                innovation = [input_node.id, output_node.id]
        else:
            input_node = input_output_nodes[0]
            output_node = input_output_nodes[1]
            if [input_node.id, output_node.id] in self.innovations_array or [output_node.id, input_node.id] in self.innovations_array:
                pass
            else:
                innovation = [input_node.id, output_node.id]

        if innovation is not None:
            self.innovations_array.append(innovation)
            current_innovation_index = len(self.innovations_array)
            connection = Connection(input_node.id, output_node.id, current_innovation_index)
            self.connections.append(connection)
            return connection

        return None

    def select_random_nodes(self):
        first_node = self.nodes[np.random.randint(0, len(self.nodes))]
        second_node = self.nodes[np.random.randint(0, len(self.nodes))]
        while first_node.id == second_node.id or (first_node.node_type != Node.TYPE_HIDDEN and first_node.node_type == second_node.node_type):
            second_node = self.nodes[np.random.randint(0, len(self.nodes))]
        if first_node.node_type == Node.TYPE_OUTPUT:
            return [second_node, first_node]
        elif second_node.node_type == Node.TYPE_INPUT:
            return [second_node, first_node]
        return [first_node, second_node]

    def select_random_connection(self):
        connection = self.connections[np.random.randint(0, len(self.connections))]
        return connection

    def get_connection_by_input_output_node_id(self, input_node_id, output_node_id):
        for index, connection in enumerate(self.connections):
            if connection.from_node == input_node_id and connection.to_node == output_node_id:
                return connection
            elif connection.from_node == output_node_id and connection.to_node == input_node_id:
                return connection
        return None

    def create_node(self):
        node = Node(len(self.nodes), Node.TYPE_HIDDEN)
        random_nodes = self.select_random_nodes()
        connection_one = [random_nodes[0], node]
        connection_two = [node, random_nodes[1]]
        self.create_connection(connection_one)
        self.create_connection(connection_two)
        old_connection = self.get_connection_by_input_output_node_id(random_nodes[0].id, random_nodes[1].id)
        if old_connection is None:
            old_connection = self.get_connection_by_input_output_node_id(random_nodes[1].id, random_nodes[0].id)
        if old_connection is not None:
            old_connection.status = 0
        self.nodes.append(node)

    def enable_disable_connection(self):
        connection = self.select_random_connection()
        if connection.status == 0:
            # print("# enabling connection innovation id", connection.innovation_id, "#")
            connection.status = 1
        else:
            # print("# disabling connection innovation id", connection.innovation_id, "#")
            connection.status = 0

    def adjust_weight(self):
        connection = self.select_random_connection()
        # print("# adjusting weight of connection innovation id", connection.innovation_id, "#")
        # print("Old weight:", connection.weight)
        connection.weight += np.random.uniform(-1, 1)
        # print("New weight:", connection.weight)

    def change_weight(self):
        connection = self.select_random_connection()
        # print("# changing weight of connection innovation id", connection.innovation_id, "#")
        # print("Old weight:", connection.weight)
        connection.weight = np.random.uniform(-1, 1)
        # print("New weight:", connection.weight)

    def get_node_by_id(self, node_id):
        return self.nodes[node_id]

    def get_weights_by_node(self, node_id):
        node = self.get_node_by_id(node_id)
        weights = []
        for connection in self.connections:
            if connection.to_node == node.id and connection.status == Connection.ENABLED:
                weights.append(connection.weight)

        return weights

    def get_input_connections_by_node(self, node):
        node = self.get_node_by_id(node.id)
        connections = []
        for connection in self.connections:
            if connection.to_node == node.id and connection.status == Connection.ENABLED:
                connections.append(connection)

        return connections

    def fit(self, input_data):
        self.outputs = []
        for index, input_node in enumerate(self.input_nodes):
            input_node.output = input_data[index]

        for output_node in self.output_nodes:
            output_node_connections = self.get_input_connections_by_node(output_node)
            output_node.input = self.calculate_output_for_nodes(output_node_connections)
            output_node.output = output_node.activation_function(output_node.input + output_node.bias)
            self.outputs.append(output_node.output)

        return self.outputs

    def print_output(self):
        print("Output:", self.outputs)

    def calculate_output_for_nodes(self, node_connections):
        sum_value = 0.0
        for connection in node_connections:
            node = self.get_node_by_id(connection.from_node)
            if node.output is None:
                node_connections = self.get_input_connections_by_node(node)
                node.input = self.calculate_output_for_nodes(node_connections)
                node.output = node.activation_function(node.input + node.bias)
            sum_value += node.output * connection.weight
        return sum_value

    def __str__(self):
        to_print = "Model id: " + str(self.model_id) + "\nNodes:\n"
        for node in self.nodes:
            to_print += node.__str__()+"\n"
        to_print += "Connections:\n"
        for connection in self.connections:
            to_print += connection.__str__()+"\n"
        to_print += "Innovations:\n"
        for innovation in self.innovations_array:
            to_print += str(innovation)
        to_print += "\n"
        return to_print
