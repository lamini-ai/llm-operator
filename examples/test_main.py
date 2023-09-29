import os

from examples.test_motivation import MotivationOperator
from examples.test_onboarding import OnboardingOperator
from llm_operator import Operator
os.environ["LLAMA_ENVIRONMENT"] = "PRODUCTION"


class MainApp(Operator):
    def callOnboardingOperator(self, message: str):
        """
        call the onboarding operator. it has operations like set user age, height, weight, etc.

        Parameters:
        message: user input message.
        """
        print("callOnboardingOperator: ")
        onboard_op = OnboardingOperator("OnboardingOperator", "examples/models/clf/OnboardingOperator")
        return onboard_op(message)

    def callMotivationOperator(self, message: str):
        """
        call the motivation operator. it has operations like send congratulatory message, motivational message, etc.

        Parameters:
        message: user input message.
        """
        print("callMotivationOperator: ")
        motivate_op = MotivationOperator("MotivationOperator", "examples/models/clf/MotivationOperator")
        return motivate_op(message)

    def __call__(self, mssg):
        self.add_operation(self.callOnboardingOperator)
        self.add_operation(self.callMotivationOperator)
        return self.run(mssg)


if __name__ == '__main__':
    agent = MainApp("MainApp", "examples/models/clf/MainApp")
    query = "Yay! you did so well today. Great workout!"
    response = agent(query)
    print(response)

