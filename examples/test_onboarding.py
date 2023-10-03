import os
import argparse

from llm_operator import Operator

os.environ["LLAMA_ENVIRONMENT"] = "PRODUCTION"


class OnboardingOperator(Operator):
    def __init__(self):
        super().__init__()
        self.add_operation(self.setAge)
        self.add_operation(self.setHeight)

    def setAge(self, age: int):
        """
        set the age of a person

        Parameters:
        age: age of the person in years.
        """
        print("It is indicated to be the age of the user.")
        return f"Age has been set. Age= {age}"

    def setHeight(self, height: int, units: str):
        """
        set the height of a person

        Parameters:
        height: height of the person in numbers.
        units: units of the height like feet, inches, cm, etc.
        """
        print("It is indicated to be the height of the user.")
        return f"Height has been set. Height={height}, units={units}"


def train(operator_save_path, training_data=None):
    """Trains the Operator."""
    operator = OnboardingOperator()
    operator.train(training_data, operator_save_path)
    print('Done training!')

def inference(queries, operator_save_path):
    operator = OnboardingOperator().load(operator_save_path)
    
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
        default="examples/models/OnboardingOperator/",
    )

    parser.add_argument(
        "--training_data",
        type=str,
        help="Path to dataset (CSV) to train on. Optional.",
        default="examples/data/onboarding.csv",
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
    
    default_queries = ["who me? I am of age fifty nine, my friend.", "I am 6 feet tall."]
    queries = args.query if args.query else default_queries
    inference(queries, args.operator_save_path)

if __name__ == '__main__':
    main()
