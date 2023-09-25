import os
from DAOperator import DecisionOperator
from test_onboarding import OnboardingOperator
from test_motivation import MotivationOperator
from llm_operator import Operation, Operator


class App(DecisionOperator):
    def __init__(self):
        self.op1 = Operation(name="startOnboarding")
        self.op2 = Operation(name="startMotivation")
        self.add_operations(self, [self.op1, self.op2])
        # app.run()

    def startOnboarding(self):
        print("inside startOnboarding")
        self.op1.description = "invoke onboarding agent to allow user to onboard"
        # ob = OnboardingOperator()
        # return ob.run()

    def startMotivation(self):
        print("inside startMotivation")
        self.op2.description = "invoke motivation agent to send motivation messages to the user"
        # ob = MotivationOperator()
        # return ob.run()


if __name__ == '__main__':
    app = App()
