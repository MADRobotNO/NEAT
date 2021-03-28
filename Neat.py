from NeuralNetwork import Model


class Neat:

    def __init__(self, number_of_inputs, number_of_outputs, number_of_models):
        self.all_models = []
        self.genomes = []
        self.number_of_models = number_of_models
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.initialize_models()

    def initialize_models(self):
        for i in range(0, self.number_of_models):
            if i == 0:
                model = Model(self.number_of_inputs, self.number_of_outputs, len(self.all_models))
                self.all_models.append(model)
            else:
                model = Model(self.number_of_inputs, self.number_of_outputs, len(self.all_models))
                model.mutate()
                self.all_models.append(model)

    def fit_models(self, input_data):
        for model in self.all_models:
            for input_dat in input_data:
                result = model.fit(input_dat)
                print("Input data:", input_dat)
                print(model)
                print("### Results", result, "###\n")

    def mutate_models(self):
        for model in self.all_models:
            model.mutate()

    def get_model_by_id(self, model_id):
        return self.all_models[model_id]

    def __str__(self):
        to_print = ""
        for index, model in enumerate(self.all_models):
            to_print += "Model nr." + str(index) + "\n"
            to_print += model.__str__() + "\n"
        return to_print
