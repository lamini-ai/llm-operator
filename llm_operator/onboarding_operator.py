import os
import argparse

from base_operator import Operator

os.environ["LLAMA_ENVIRONMENT"] = "STAGING"


class OnboardingOperator(Operator):
    def __init__(self):
        super().__init__()
        self.add_operation(self.setDateOfBirth)
        self.add_operation(self.setGender)
        self.add_operation(self.setWeight)

    def setGender(self, gender: str):
        """
        set the gender of the person. If it is unclear if 'Male' or 'Female', then set it as 'Unspecified'.

        Parameters:
        gender: gender of the person. Either 'Male', 'Female' or 'Unspecified'.
        """
        print("It is indicated to be the age of the user.")
        return f"Gender has been set. DOB= {gender}"

    def setDateOfBirth(self, dob: str):
        """
        set the date of birth of a person. If date of birth is unclear, set it as 0000-00-00. If no year is given, assume it is 0000.

        Parameters:
        dob: date  of birth in ISO date format.
        """
        print("It is indicated to be the age of the user.")
        return f"DOB has been set. DOB= {dob}"

    def setWeight(self, weight: int, units: str):
        """
        set the weight of the person. if no weight is given, set it as 0. If no units are given, assume it is in lbs.

        Parameters:
        weight: height of the person in numbers.
        units: units of the height like feet, inches, cm, etc.
        """
        print("It is indicated to be the height of the user.")
        return f"Weight has been set. Height={weight}, units={units}"


def train(operator_save_path, training_data=None):
    """Trains the Operator."""
    operator = OnboardingOperator()
    print(training_data)
    operator.train(operator_save_path, training_data)
    print("Done training!")


def inference(queries, operator_save_path):
    operator = OnboardingOperator().load(operator_save_path)

    for query in queries:
        print(f"\n\nUser message: {query}")
        response = operator(query)
        print(response)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--operator_save_path",
        type=str,
        help="Path to save the operator / use the saved operator.",
        default="models/OnboardingOperator/",
    )

    parser.add_argument(
        "--training_data",
        type=str,
        help="Path to dataset (CSV) to train on. Optional.",
        default="data/onboarding.csv",
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

    parser.add_argument(
        "-l", action="store_true", help="this flag is a no-op to silence errors"
    )

    args = parser.parse_args()

    if args.operator_save_path[-1] != "/":
        args.operator_save_path += "/"

    if args.train:
        train(args.operator_save_path, args.training_data)

    default_queries = [
        "I consider myself a man.",
        "I was born on the Winter Solstice of 1980.",
        "I weigh 150 stones",
        "I am Genderfluid.",
    ]
    queries = args.query if args.query else default_queries
    inference(queries, args.operator_save_path)


if __name__ == "__main__":
    main()
