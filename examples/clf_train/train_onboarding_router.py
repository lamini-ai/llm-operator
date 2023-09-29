import argparse

from operator_classifier import RoutingOperator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='Name of the Operator')
    # This argument is optional and can be skipped. However, it is advised to train with some high quality data for optimal performance.
    parser.add_argument('--training_file', default=None,
                        help='csv file containing labeled data examples for each class.')
    parser.add_argument('--classes_file', help='json file which contains class names and their prompts.')
    parser.add_argument('--output_folder',
                        help='path of folder where to save the model. Saved by "<router_name>.lamini."')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')

    args = parser.parse_args()
    router_name = args.name
    training_file = args.training_file
    classes_file = args.classes_file
    output_folder = args.output_folder
    verbose_mode = args.verbose

    clf = RoutingOperator(router_name, output_folder)
    clf.fit(classes_file, training_file)
    resp = clf.predict(['I am 6 ft tall', 'I am 10 years old'])
    print(resp)

'''
python3 examples/clf_train/train_onboarding_router.py 
--name OnboardingOperator 
--training_file examples/models/clf/OnboardingOperator/train_clf.csv 
--classes_file examples/models/clf/OnboardingOperator/clf_classes_prompts.json 
--output_folder examples/models/clf/OnboardingOperator/
'''
