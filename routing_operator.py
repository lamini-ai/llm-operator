from lamini import LaminiClassifier
import os
import pandas as pd
import json


class RoutingOperator:
    def __init__(self, model_load_path):
        self.model_load_path = model_load_path
        if not os.path.exists(self.model_load_path):
            self.classifier = LaminiClassifier()
        else:
            self.classifier = LaminiClassifier.load(self.model_load_path)

    def __add_data(self, classes, training_data_path):
        '''
        Gets names and prompts of classes for routing.

        classes: list of class names for classification by router
        training_data_path: json file containing the name and prompt of each class
        '''
        df = pd.read_csv(training_data_path, quotechar='"', )
        for cl in classes:
            d = df.loc[df['class_name'] == cl]['data'].to_list()
            self.classifier.add_data_to_class(cl, d)

    def fit(self, classes_dict, training_data_path = None):
        '''
        to train/prompt-train the routing classifier.

        classes_dict: dict containing name of class and prompt for the class
        training_data_path: optional string path of training data csv.
        '''
        if training_data_path:
            self.__add_data(classes_dict, training_data_path)
        self.classifier.prompt_train(classes_dict)

    def save(self, model_save_path):
        print("Saving router to:", model_save_path)
        self.classifier.save(model_save_path)

    def predict(self, data):
        '''
        Predict label and probabilities

        data: list of strings to predict
        Output format: tuple of 2 lists.
        List 1 of len(data): predicted label of every query string.
        List 2 of len(data): probability distribution of each label for every query string.
        '''
        prediction = self.classifier.predict(data)
        probabilities = self.classifier.predict_proba(data)
        return prediction, probabilities
