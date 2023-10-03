import pandas as pd
from lamini import LlamaClassifier


class LLMRoutingAgent:
    def __init__(self, model_load_path):
        self.model_load_path = model_load_path
        self.ROUTING_THRESHOLD = 0.4

    def __load_clf(self, cl):
        '''
        Load the classifier from the model_save_path
        '''
        save_path = f"{self.model_load_path}/{cl}.pkl"
        classifier = LlamaClassifier.load(save_path)
        return classifier

    def __get_negative_data(self, pos_len, d):
        each_size = int(pos_len // 1.2 * len(set(d["class_name"])))
        filtered_data = []
        grouped = d.groupby("class_name")["data"].agg(list)
        print(f"No. of negative classes for this operation: {len(grouped)}")
        print(f"Data size from each class: {each_size}")
        for class_name, group_data in grouped.items():
            filtered_data.extend(group_data[:each_size])
        return filtered_data

    def train_with_data(self, classes_dict, training_data_path):
        df = pd.read_csv(training_data_path, quotechar='"')
        for cl in classes_dict:
            classifier = LlamaClassifier()
            positive_data = df.loc[df['class_name'] == cl]['data'].to_list()
            print(f"\nNo. of positive {cl} class samples: {len(positive_data)}")
            classifier.add_data_to_class(cl, positive_data)
            negative_data = self.__get_negative_data(len(positive_data), df.loc[df['class_name'] != cl])
            classifier.add_data_to_class("not" + cl, negative_data)
            classifier.train()
            save_path = f"{self.model_load_path}/{cl}.pkl"
            classifier.save(save_path)

    def fit(self, classes_dict, training_data_path = None):
        '''
        to train/prompt-train the routing classifier.

        classes_dict: dict containing name of class and prompt for the class
        training_data_path: optional string path of training data csv.
        '''

        print("Training operator...")
        if training_data_path:
            self.train_with_data(classes_dict, training_data_path)

    def predict(self, data, classes_dict):
        '''
        Predict label and probabilities

        data: list of strings to predict
        Output format: tuple of 2 lists.
        List 1 of len(data): predicted label of every query string.
        List 2 of len(data): probability distribution of each label for every query string.
        '''
        data_labels = []
        for d in data:
            pt = []
            for cl, _ in classes_dict.items():
                clf = self.__load_clf(cl)
                print(f"\n{d}: {cl} {clf.predict([d])} {clf.predict_proba([d])}")
                proba = clf.predict_proba([d])[0][0]
                if proba >= self.ROUTING_THRESHOLD:
                    pt.append(cl)
            data_labels.append(pt)

        return data_labels
