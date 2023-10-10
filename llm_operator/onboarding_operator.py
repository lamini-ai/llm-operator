import os
import argparse
from typing import Optional
from base_operator import Operator
from base_planning_operator import PlanningOperator
from base_inquiry_operator import InquiryOperator
from llama import BasicModelRunner
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
        return f"{operator_response}"

    def setAge(self, age: int):
        """
        given the age of a person, set their age in the system.

        Parameters:
        age: age of the person in years.
        """
        print("It is indicated to be the age of the user.")
        self.age = age
        return f"Age has been set. Age= {age}"

    def setHeight(self, height: int):
        """
        given the height of a person, set their height in the system.

        Parameters:
        height: height of the person in cms. If the height is in feet and inches, convert it to cm. Eg: 5'6" = 167.
        """
        self.height = height
        print("It is indicated to be the height of the user.")
        return f"Height has been set. Height={height}"

    def setWeight(self, weight: int):
        """
        given the weight of a person, set their weight in the system.

        Parameters:
        weight: weight of the person in pounds. If the weight is in any other unit, convert it to lbs.
        """
        self.weight = weight
        print("It is indicated to be the weight of the user.")
        return f"Weight has been set. Weight={weight}"

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
    system_prompt = """The system asks a question to get user details. Use the use input to decide which tool to use for utilising the user information. Ask clarifying questions if something is unclear.
Example session:
System: Enter your weight, height and age.
User: my age is 50.
System: [PLAN] 1. Calling setAge(50) as user has provided 50 as his age."""

    planning_prompt = "Use the latest user message alongwith chat history to decide which tool/tools to use to act on user information."

    operator = OnboardingOperator(model_name=None,
                                  system_prompt=system_prompt,
                                  planning_prompt=planning_prompt,
                                  verbose=args.verbose
                                  ).load(operator_save_path)

    query = "Enter your age (in years)."
    chat_history = f"System: {query}\n"
    while not operator.age:
        user_response = input(query)
        chat_history += f"User: {user_response}\n"
        operator_resp_followup_query = operator(user_response, chat_history)
        print(operator_resp_followup_query)
        chat_history += f"System: {operator_resp_followup_query}\n"
        query = operator_resp_followup_query

    query = "Enter your height (in cm)."
    chat_history = f"System: {query}\n"
    while not operator.height:
        user_response = input(query)
        chat_history += f"User: {user_response}\n"
        operator_resp_followup_query = operator(user_response, chat_history)
        print(operator_resp_followup_query)
        chat_history += f"System: {operator_resp_followup_query}\n"
        query = operator_resp_followup_query

    query = "Enter your weight (in lbs)."
    chat_history = f"System: {query}\n"
    while not operator.weight:
        user_response = input(query)
        chat_history += f"User: {user_response}\n"
        operator_resp_followup_query = operator(user_response, chat_history)
        print(operator_resp_followup_query)
        chat_history += f"System: {operator_resp_followup_query}\n"
        query = operator_resp_followup_query
    # while not operator.isDone():
    #     user_response = input(query)
    #     chat_history += f"User: {user_response}\n"
    #     operator_resp_followup_query = operator(user_response, chat_history)
    #     print(operator_resp_followup_query)
    #     if "has been set" in operator_resp_followup_query:
    #         chat_history = ""
    #     chat_history += f"System: {operator_resp_followup_query}\n"
    #     query = operator_resp_followup_query

if __name__ == '__main__':
    main()
