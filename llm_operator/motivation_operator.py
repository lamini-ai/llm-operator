import os
import argparse

from base_operator import Operator

os.environ["LLAMA_ENVIRONMENT"] = "PRODUCTION"


class MotivationOperator(Operator):
    def __init__(self):
        super().__init__()

        self.add_operation(self.setReminder)
        self.add_operation(self.sendCongratsMessage)
        self.add_operation(self.sendFollowupMessage)

    def setReminder(self, workout_name: str, workout_time: str):
        """
        set a reminder message to the user to do workout.

        Parameters:
        workout_name: name of the workout. if no name given, keep it static at 'no-name'
        workout_time: date and time to schedule the workout.
        """
        print("It is indicated to be a reminder message.")
        return f"Reminder has been set. Workout: {workout_name}, Time: {workout_time}"

    def sendCongratsMessage(self, message: str):
        """
        send a congratulatory message to the user on completing the workout.

        Parameters:
        message: the congratulatory message.
        """
        print("It is indicated to be a congratulatory message.")
        return "Sending user message=" + message

    def sendFollowupMessage(self, message: str):
        """
        send a follow-up message to the user checking on him for missing the workout.

        Parameters:
        message: a message meant to follow up with the user on missing a workout
        """
        print("It is indicated to be a follow up message.")
        return "Sending user message=" + message


def train(operator_save_path, training_data=None):
    """Trains the Operator."""
    operator = MotivationOperator()
    operator.train(operator_save_path, training_data)
    print('Done training!')

def inference(queries, operator_save_path):
    operator = MotivationOperator().load(operator_save_path)
    
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
        default="models/MotivationOperator/",
    )

    parser.add_argument(
        "--training_data",
        type=str,
        help="Path to dataset (CSV) to train on. Optional.",
        default="data/motivation.csv",
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
        "Yay you did it. This is awesome!",
        "Hey Aaron, hope you are well! I noticed you missed our workout together at Hike in Mt. Abby, Alaska on Monday. It is important to stay consistent with your fitness routine, so I hope you can make it to our next workout together."
    ]
    queries = args.query if args.query else default_queries
    inference(queries, args.operator_save_path)


if __name__ == '__main__':
    main()