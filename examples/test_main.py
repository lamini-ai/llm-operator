import os
import argparse

from examples.test_motivation import MotivationOperator
from examples.test_onboarding import OnboardingOperator
from llm_operator import Operator

os.environ["LLAMA_ENVIRONMENT"] = "PRODUCTION"


class MainApp(Operator):
    def __init__(self):
        super().__init__()
        
        self.onboarding_operator_save_path = "examples/models/clf/OnboardingOperator/"
        self.onboarding_operator = OnboardingOperator().load(self.onboarding_operator_save_path)

        self.motivation_operator_save_path = "examples/models/clf/MotivationOperator/"
        self.motivation_operator = MotivationOperator().load(self.motivation_operator_save_path)
    
        self.add_operation(self.call_onboarding_operator)
        self.add_operation(self.call_motivation_operator)

    def call_onboarding_operator(self, message: str):
        """
        call the onboarding operator. it has operations like set user age, height, weight, etc.

        Parameters:
        message: user input message.
        """
        print("\nIt is indicated that the user is new and needs to be onboarded.")
        print("call_onboarding_operator...\n")
        return self.onboarding_operator(message)

    def call_motivation_operator(self, message: str):
        """
        call the motivation operator. it has operations like send congratulatory message, motivational message, etc.

        Parameters:
        message: user input message.
        """
        print("\nIt is indicated that this meant to be a motivational message.")
        print("call_motivation_operator...\n")
        return self.motivation_operator(message)


def train(operator_save_path, training_data=None):
    """Trains the Operator."""
    operator = MainApp()
    operator.train(training_data, operator_save_path)
    print('Done training!')

def inference(queries, operator_save_path):
    operator = MainApp().load(operator_save_path)
    
    for query in queries:
        print(f"\n\nQuery: {query}")
        response = operator(query)
        print(response)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--operator_save_path",
        type=str,
        help="Path to save the operator / use the saved operator.",
        default="examples/models/clf/MainApp/",
    )

    parser.add_argument(
        "--training_data",
        type=str,
        help="Path to dataset (CSV) to train on. Optional.",
        default=None,
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model.",
        default=False,
    )

    parser.add_argument(
        "--query",
        type=str,
        nargs="+",
        action="extend",
        help="Queries to run",
        default=[],
    )

    args = parser.parse_args()

    if args.operator_save_path[-1] != "/":
        args.operator_save_path += "/"

    if args.train:
        train(args.operator_save_path, args.training_data)
    
    default_queries = [
        "You missed your workout yesterday. Just wanted to check in!",
        "Hey Aaron, hope you are well! I noticed you missed our workout together at Hike in Mt. Abby, Alaska on Monday. It is important to stay consistent with your fitness routine, so I hope you can make it to our next workout together.", 
        "I am 6 feet tall.",
    ]
    queries = args.query if args.query else default_queries
    inference(queries, args.operator_save_path)

if __name__ == '__main__':
    main()


