from lamini import LaminiClassifier
import os
import pandas as pd
import json


class RoutingOperator:
    def __init__(self, model_load_path, classes_load_path):
        self.classes = self.__get_classes(classes_load_path)
        self.model_save_path = model_load_path

        if not os.path.exists(model_load_path):
            self.classifier = LaminiClassifier()
        else:
            self.classifier = LaminiClassifier.load(self.model_save_path)

    def __get_classes(self, classes_path: str):
        '''
        Gets names and prompts of classes for routing.

        classes_path: json file containing the name and prompt of each class
        '''
        with open(classes_path, "r") as json_file:
            classes = json.load(json_file)
        return classes

    def __add_data(self, training_data_path):
        '''
        Gets names and prompts of classes for routing.

        training_data_path: json file containing the name and prompt of each class
        '''
        df = pd.read_csv(training_data_path, quotechar='"', )
        for cl in self.classes:
            d = df.loc[df['class_name'] == cl]['data'].to_list()
            self.classifier.add_data_to_class(cl, d)

    def fit(self, training_data_path = None):
        '''
        to train/prompt train the routing classifier.

        training_data_path: optional string path of training data csv.
        '''
        if training_data_path:
            self.__add_data(training_data_path)
        self.classifier.prompt_train(self.classes)
        self.classifier.save(self.model_save_path)

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


if __name__ == '__main__':
    m = "/Users/ayushisharma/Documents/operator-prototype/models/clf/MotivationOperator/MotivationOperator.lamini"
    c = "/Users/ayushisharma/Documents/operator-prototype/models/clf/MotivationOperator/cls.json"
    t = "/Users/ayushisharma/Documents/operator-prototype/models/clf/MotivationOperator/train_clf.csv"
    clf = RoutingOperator(m, c)
    # clf.fit(t)
    resp = clf.predict(['You did it! that was awesome.', 'Set a workout for Nov 10 5pm'])
    print(resp)
