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

    def add_operations(self):
        self.add_operation(self.setAge)
        self.add_operation(self.setHeight)

    def __call__(self, mssg):
        self.add_operation(self.setAge)
        self.add_operation(self.setHeight)
        return self.run(mssg)

def main():
    # train and  inference
    # #optional training file path
    # training_file = None
    # operator_save_path = "examples/models/clf/OnboardingOperator/"
    # operator = OnboardingOperator()
    # operator.add_operations()
    # operator.train(training_file, operator_save_path)
    # query = "who me? I am of age fifty nine, my friend."
    # response = operator(query)

    # inference
    operator_save_path = "examples/models/clf/OnboardingOperator/router.pkl"
    operator = OnboardingOperator().load(operator_save_path)
    operator.add_operations()

    query2 = "who me? I am of age fifty nine, my friend."
    print(f"\n\nQuery: {query2}")
    response2 = operator(query2)
    print(response2)
    query3 = "I am 6 feet tall."
    print(f"\n\nQuery: {query3}")
    response3 = operator(query3)
    print(response3)


if __name__ == '__main__':
    main()
