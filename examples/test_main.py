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
        print("\nIt is indicated that the user is new and needs to be onboarded.")
        print("callOnboardingOperator...\n")
        operator_save_path = "examples/models/clf/OnboardingOperator/router.pkl"
        operator = OnboardingOperator().load(operator_save_path)
        operator.add_operations()
        return operator(message)

    def callMotivationOperator(self, message: str):
        """
        call the motivation operator. it has operations like send congratulatory message, motivational message, etc.

        Parameters:
        message: user input message.
        """
        print("\nIt is indicated that this meant to be a motivational message.")
        print("callMotivationOperator...\n")
        operator_save_path = "examples/models/clf/MotivationOperator/router.pkl"
        operator = MotivationOperator().load(operator_save_path)
        operator.add_operations()
        return operator(message)

    def add_operations(self):
        self.add_operation(self.callOnboardingOperator)
        self.add_operation(self.callMotivationOperator)

    def __call__(self, mssg):
        return self.run(mssg)


if __name__ == '__main__':
    # train and  inference
    # #optional training file path
    # training_file = None
    # operator_save_path = "examples/models/clf/MainApp/"
    # operator = MainApp()
    # operator.add_operations()
    # operator.train(training_file, operator_save_path)
    # query = "You missed your workout yesterday. Just wanted to check in!"
    # response = operator(query)

    # inference
    operator_save_path = "examples/models/clf/MainApp/router.pkl"
    operator = MainApp().load(operator_save_path)
    operator.add_operations()

    query2 = "Hey Aaron, hope you are well! I noticed you missed our workout together at Hike in Mt. Abby, Alaska on Monday. It is important to stay consistent with your fitness routine, so I hope you can make it to our next workout together."
    print(f"\n\nQuery: {query2}")
    response2 = operator(query2)
    print(response2)
    query3 = "I am 6 feet tall."
    print(f"\n\nQuery: {query3}")
    response3 = operator(query3)
    print(response3)


