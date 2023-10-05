import os
import argparse

from base_planning_operator import PlanningOperator


os.environ["LLAMA_ENVIRONMENT"] = "PRODUCTION"
os.environ["LLAMA_ENVIRONMENT"] = "STAGING"


class PlanningMotivationOperator(PlanningOperator):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)

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


def main():

    args = argparse.ArgumentParser()
    args.add_argument("--verbose", action="store_true", help="Print extra info", default=False)
    args = args.parse_args()

    operator_save_path = "models/MotivationOperator/"
    operator = PlanningMotivationOperator(verbose=args.verbose).load(operator_save_path)
    
    chat_history = """User: Hi, I'm feeling down
System: I'm sorry to hear that. What would you like to do?"""
    query = "I want to do a workout to feel better"
    response = operator(query, chat_history)
    print(response)


if __name__ == '__main__':
    main()