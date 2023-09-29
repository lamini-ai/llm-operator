import argparse

from routing_operator import RoutingOperator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='Name of the Operator')

    # This argument is optional and can be skipped. However, it is advised to train with some high quality data for
    # optimal performance.
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

    rop = RoutingOperator(router_name, classes_file, output_folder)
    # clf.fit()
    queries = ['Wake me up at 10am', 'Where have you been? You missed your workout.', 'Schedule my "Run and burn" workout at 2pm tomorrow', 'Wow, look at you go! That was awesome.']
    resp = rop.predict(queries)
    print(resp)

'''
python3 train_router.py 
--name MainApp 
--training_file examples/models/clf/MainApp/train_clf.csv 
--classes_file examples/models/clf/MainApp/clf_classes_prompts.json 
--output_folder examples/models/clf/MainApp/

python3 train_router.py 
--name MotivationOperator 
--training_file examples/models/clf/MotivationOperator/train_clf.csv 
--classes_file examples/models/clf/MotivationOperator/clf_classes_prompts.json 
--output_folder examples/models/clf/MotivationOperator/
'''
