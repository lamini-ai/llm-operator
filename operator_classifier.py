from lamini import LaminiClassifier
import os


class RoutingOperator:
    def __init__(self, classes, model_load_path):
        if not os.path.exists(model_load_path):
            self.classifier = LaminiClassifier()
            self.classify(classes, model_load_path)
        else:
            self.classifier = LaminiClassifier.load(model_load_path)

    def classify(self, classes, path):
        self.classifier.prompt_train(classes)
        self.classifier.save(path)

    def predict(self, data):
        prediction = self.classifier.predict(data)
        probabilities = self.classifier.predict_proba(data)
        return prediction, probabilities
