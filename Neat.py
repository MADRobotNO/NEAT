from NeuralNetwork import Model


model = Model(3, 2, True)

# print("Nodes")
# for node in model.nodes:
#     print(node)

print("\nConnections")
for connection in model.connections:
    print(connection)

print("\nInput Nodes")
for node in model.input_nodes:
    print(node)
print("\nOutput Nodes")
for node in model.output_nodes:
    print(node)
