import os
from llm_operator import Operator
from llama import Lamini

os.environ["LLAMA_ENVIRONMENT"] = "PRODUCTION"


class OnboardingOperator:
    def __init__(self):
        self.person_age = 0
        self.person_height = 0

    def get_operation(self):
        prompt_template = """<s>[INST] <<SYS>>
        Respond to the message using function calls. You have access to the following tools:
        <</SYS>>
        {input:operations}

        Use the following format:

        Message: the input message you must use
        Tool: the tool you will use. Only write the name of the tool

        Begin!

        Message: {input:query}
        Tool: """

        operation_selector = Lamini(
            "operator", "meta-llama/Llama-2-7b-chat-hf", prompt_template
        )
        return operation_selector

    def generate_args(self):
        prompt_template = """Given:
        {input:query.field} ({input:query.context}): {input:query}
        {input:operation.field} ({input:operation.context}): {input:operation}
        Generate:
        {output:age.field} after "{output:age.field}:"
        {output:age.field}: """
        argument_generator = Lamini(
            "operator", "meta-llama/Llama-2-7b-chat-hf", prompt_template
        )
        return argument_generator

    def generate_response(self):
        prompt_template = """\
        You are a helpful assistant. You've just been asked to help with a task with the tools:
        {input:operations}
        The user's message: {input:query}
        You decided to use the tool {input:operation} with the arguments {input:args}
        Once you used the tool you got the output {input:output}
        Respond to the user's message 

        {input:query}

        with a final message explaining the actions you took. Do not talk about using tools. Be helpful and inform the user of what has happened.
        If the tool's output includes an error, tell the user that there was an error. 
        Otherwise, acknowledge the user's message and tell them what you did: 
        """
        vocal_llm = Lamini("vocal", "meta-llama/Llama-2-13b-chat-hf", prompt_template)
        return vocal_llm

    def setAge(self, age: int):
        """set the age of a person"""
        self.person_age = age

    def setHeight(self, height: int):
        """set the height of a person in inches"""
        self.person_height = height

    def __call__(self, mssg):
        # selects which tool to call
        op = self.get_operation()

        # generates input args for the tool
        args = self.generate_args()

        # generates final response of the tool
        message = self.generate_response()
        operator = Operator(op, args, message)
        operator.add_operation(self.setAge)
        operator.add_operation(self.setHeight)
        return operator.run(mssg)


if __name__ == '__main__':
    agent = OnboardingOperator()
    response = agent("I am 19 years old and 6ft tall.")
    # print(response)
    #
    # print("FINAL OUTCOMES: ")
    # print("age:", agent.person_age)
    # print("height:", agent.person_height)
