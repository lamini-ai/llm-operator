import os
import argparse
from typing import Optional
from base_operator import Operator
from base_planning_operator import PlanningOperator
from base_inquiry_operator import InquiryOperator

os.environ["LLAMA_ENVIRONMENT"] = "PRODUCTION"


class OnboardingOperator(InquiryOperator):
    def __init__(self, model_name: Optional[str] = None, system_prompt: Optional[str] = None,
                 planning_prompt: Optional[str] = None, verbose=False):
        super().__init__(model_name, system_prompt, planning_prompt, verbose=verbose)
        self.age = None
        self.height = None
        self.weight = None
        self.add_operation(self.setAge)
        self.add_operation(self.setHeight)
        self.add_operation(self.setWeight)
        self.add_operation(self.clarify)

    def isDone(self):
        return self.age is not None and self.height is not None and self.weight is not None

    def clarify(self, operator_response: str):
        """
        clarify user input by asking more questions or replying to their question and asking to complete onboarding tasks of setting weight, height and age.

        Parameters:
        operator_response: response from the operator to user input.
        """
        print(f"\n\n[Clarifying user input.] {operator_response}")

    def setAge(self, age: int):
        """
        given the age of a person, set their age in the system.

        Parameters:
        age: age of the person in years.
        """
        print("It is indicated to be the age of the user.")
        self.age = age
        return f"Age has been set. Age= {age}"

    def setHeight(self, height: int, units: str):
        """
        given the height of a person, set their height in the system.

        Parameters:
        height: height of the person in numbers.
        units: units of the height like feet, inches, cm, etc.
        """
        self.height = height
        print("It is indicated to be the height of the user.")
        return f"Height has been set. Height={height}, units={units}"

    def setWeight(self, weight: int, units: str):
        """
        given the weight of a person, set their weight in the system.

        Parameters:
        weight: weight of the person in numbers.
        units: units of the weight like lbs, kgs, stones, etc.
        """
        self.weight = weight
        print("It is indicated to be the weight of the user.")
        return f"Weight has been set. Weight={weight}, units={units}"
    #
    # def getRecommendation(self):
    #     """
    #     suggest a workout to do for the user.
    #
    #     """
    #     print("It is indicated to recommend a workout to the user.")
    #     workout_name = "Run&Burn"
    #     return f"Suggesting workout {workout_name} for the user."
    #
    # def scheduleWorkout(self, workout_name: str, workout_time: str):
    #     """
    #     sets the workout on user schedule at the given time. if no time is given, keep it static at '01-01-2024T00:00:00'.
    #
    #     Parameters:
    #     workout_name: name of the workout to schedule.
    #     workout_time: ISO datetime to schedule the workout.
    #     """
    #     print("It is indicated to schedule the given workout for the user.")
    #     return f"Scheduled {workout_name} for the user at {workout_time}."

def train(operator_save_path, training_data=None):
    """Trains the Operator."""
    operator = OnboardingOperator()
    operator.train(operator_save_path, training_data)
    print('Done training!')

# Example session:
# System: Enter your weight, height and age.
# User: I am James. I am 40 years old and weigh 120lbs.
# System: 1. Calling setAge(40).
# 2. Calling setWeight(120, "lbs").
def main():
    args = argparse.ArgumentParser()
    args.add_argument("--verbose", action="store_true", help="Print extra info", default=False)
    args = args.parse_args()

    operator_save_path = "models/OnboardingOperator/"
    # model_name = "gpt-4"
    # train(operator_save_path)
    system_prompt = """You need to do operations to onboard a user by asking them for their height, weight and age. Ask clarifying questions until you have all the information.
Example session:
System: Enter your weight, height and age.
User: 50.
System: Plan: Calling clarify('Is that your height, weight or age?')
User: That's my age.
System: [PLAN] 1. Calling setAge(50)."""

    planning_prompt = "Based on user input, decide whether to ask clarifying questions or use a tool/tools to act on user information."

    operator = OnboardingOperator(model_name=None,
                                  system_prompt=system_prompt,
                                  planning_prompt=planning_prompt,
                                  verbose=args.verbose
                                  ).load(operator_save_path)

    query = "Enter your weight, height and age."
    chat_history = f"System: {query}\n"

    while not operator.isDone():
        user_response = input(query)
        chat_history += f"User: {user_response}\n"
        operator_resp_followup_query = operator(user_response, chat_history)
        chat_history += f"System: {operator_resp_followup_query}\n"
        if operator_resp_followup_query.endswith("Exit."):
            break
        print(operator_resp_followup_query)
        query = operator_resp_followup_query
    # chat_history = """User: Hi, I'm feeling down
    # System: I'm sorry to hear that. What would you like to do?"""
    # query = "Schedule a workout for me at 5pm today."
    # response = operator(query, chat_history)
    # print(response)
    #
    #
    # query = "I am James. I am 40 years old and weigh 120lbs."
    # response = operator(query, "")
    # print(response)


if __name__ == '__main__':
    main()
