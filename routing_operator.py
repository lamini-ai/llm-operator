from lamini import LaminiClassifier
import os
import pandas as pd
import json


class RoutingOperator:
    def __init__(self, name, classes_load_path, model_load_path):
        self.name = name
        self.classes_load_path = classes_load_path
        self.model_load_path = model_load_path
        self.ROUTING_THRESHOLD = 0.6
        # if not os.path.exists(self.model_save_path):
        #     self.classifier = LaminiClassifier()
        # else:
        #     self.classifier = LaminiClassifier.load(self.model_save_path)

    def __get_classes_dict(self, classes_path: str):
        '''
        Gets names and prompts of classes for routing.

        classes_path: json file containing the name and prompt of each class
        '''
        with open(classes_path, "r") as json_file:
            classes = json.load(json_file)
        return classes

    def __add_data(self, classifier, op_classes, training_data_path):
        '''
        Gets names and prompts of classes for routing.

        classes: list of class names for classification by router
        training_data_path: json file containing the name and prompt of each class
        '''
        df = pd.read_csv(training_data_path, quotechar='"', )
        for cl in op_classes:
            d = df.loc[df['class_name'] == cl]['data'].to_list()
            classifier.add_data_to_class(cl, d)

    def fit(self, training_data_path = None):
        '''
        to train/prompt train the routing classifier.

        classes_load_path: json file containing the name and prompt of each class
        training_data_path: optional string path of training data csv.
        '''
        operation_dict = self.__get_classes_dict(self.classes_load_path)
        for operation_name, operation_binary_prompts in operation_dict.items():
            model_save_path = f"{self.model_load_path}/{self.name}-{operation_name}.lamini"
            # positiveOperationCls = f"positive{operation_name}"
            # negativeOperationCls = f"negative{operation_name}"
            # cls = [operation_binary_prompts, negativeOperationCls]
            classifier = LaminiClassifier()
            if training_data_path:
                classifier.__add_data(classifier, operation_binary_prompts, training_data_path)
            classifier.prompt_train(operation_binary_prompts)
            classifier.save(model_save_path)

    def __load_clf(self, operation_name):
        '''
        Load the classifier from the model_save_path
        '''
        model_save_path = f"{self.model_load_path}/{self.name}-{operation_name}.lamini"
        classifier = LaminiClassifier.load(model_save_path)
        return classifier

    def predict(self, data):
        '''
        Predict label and probabilities

        data: list of strings to predict
        Output format: tuple of 2 lists.
        List 1 of len(data): predicted label of every query string.
        List 2 of len(data): probability distribution of each label for every query string.
        '''
        operation_dict = self.__get_classes_dict(self.classes_load_path)
        data_labels = []
        for d in data:
            pt = []
            for operation_name, _ in operation_dict.items():
                clf = self.__load_clf(operation_name)
                pred = clf.predict([d])[0]
                proba = clf.predict_proba([d])[0][0]
                if pred.startswith("positive") and proba > self.ROUTING_THRESHOLD:
                    pt.append(operation_name)
            data_labels.append(pt)

        return data_labels
