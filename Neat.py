from NeuralNetwork import Model
import copy
import random


class Neat:

    def __init__(self, number_of_inputs, number_of_outputs, number_of_models, random_mutation_chance):
        self.all_models = []
        self.genomes = []
        self.random_mutation_chance = random_mutation_chance
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

    def fit_models(self, input_data, targets=None, number_of_epochs=1):
        best_model_overall = None
        for epoch in range(number_of_epochs):
            for model in self.all_models:
                for index, input_dat in enumerate(input_data):
                    results = model.fit(input_dat)
                    print("### Results", results, "###\n")
                    for j, result in enumerate(results):
                        model.fitnes += (1-abs(targets[index][j]-result))
                    print(model)

            best_model_id = None
            second_best_model_id = None
            best_fitnes = 0.0
            for model in self.all_models:
                if model.fitnes > best_fitnes:
                    if best_model_id is not None:
                        second_best_model_id = best_model_id
                    best_fitnes = model.fitnes
                    best_model_id = model.model_id

            second_best_model_overall = copy.deepcopy(self.get_model_by_id(second_best_model_id))
            best_model_overall = copy.deepcopy(self.get_model_by_id(best_model_id))
            self.natural_selection(best_model_overall, second_best_model_overall)

        print("##### Best model is: ######")
        print(best_model_overall)

    def natural_selection(self, best_model_overall, second_best_model_overall):
        del self.all_models
        self.all_models = []
        best_model_overall.model_id = 0
        self.all_models.append(best_model_overall)
        second_best_model_overall.model_id = 1
        self.all_models.append(second_best_model_overall)
        for i in range(0, self.number_of_models):
            model = copy.deepcopy(best_model_overall)
            model.model_id = len(self.all_models)
            if random.random() < self.random_mutation_chance:
                model.mutate_structure()
            model.mutate_weights()
            self.all_models.append(model)

    def mutate_all_models(self):
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
