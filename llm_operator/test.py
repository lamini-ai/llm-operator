from llm_routing_agent import LLMRoutingAgent

if __name__ == '__main__':
    router_name = "OnboardingOperator"
    # training_file = "models/OnboardingOperator/train_clf.csv"
    training_file = None
    output_folder = "models/OnboardingOperator"

    rop = LLMRoutingAgent(output_folder, 0.3)
    classes_dict = {"setAge": "set the age of a person", "setHeight": "set the height of a person"}
    rop.fit(classes_dict, training_file)
    queries = ["I am 4 ft tall", "I am 45 years old", "I am 20 years old and 6 ft tall"]
    resp = rop.predict(queries, classes_dict)
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