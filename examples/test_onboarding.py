import os
from llm_operator import Operator
from llama import Lamini

os.environ["LLAMA_ENVIRONMENT"] = "PRODUCTION"


class OnboardingOperator(Operator):
    def setAge(self, age: int):
        """
        set the age of a person

        Parameters:
        age: age of the person in years.
        """
        print("setAge: ")
        return f"Age has been set. Age: {age}"

    def setHeight(self, height: int):
        """
        set the height of a person

        Parameters:
        height: height of the person in inches.
        """
        print("setHeight: ")
        return f"Height has been set. Height: {height}"

    def __call__(self, mssg):
        self.add_operation(self.setAge)
        self.add_operation(self.setHeight)
        return self.run(mssg)


if __name__ == '__main__':
    agent = OnboardingOperator("OnboardingOperator", "examples/models/clf/OnboardingOperator")
    response = agent("who me? I am of age fifty nine, my friend.")
    print(response)
    response = agent("I am 6 feet tall.")
    print(response)
