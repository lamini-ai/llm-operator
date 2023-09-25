import os
from llm_operator import Operator, Operation
# from llama import Lamini

person_age = None


class OnboardingOperator(Operator):
    def __init__(self):
        self.op1 = Operation(name="setAge")
        self.op2 = Operation(name="setHeight")
        # self.add_operations([self.op1, self.op2])

    def setAge(self):
        description = "find out the age of the user in years"
        self.op1.description = description
        self.op1.run(terminal_task = True)

    def setHeight(self, height: int):
        description = "find out the height of the user in inches"
        self.op2.description = description
        self.op2.run(terminal_task=True)
#
# os.environ["LLAMA_ENVIRONMENT"] = "STAGING"
# prompt_template = """\
# Respond to the message using function calls. You have access to the following tools:
#
# {input:operations}
#
# Use the following format:
#
# Message: the input message you must use
# Tool: the tool you will use. Only write the name of the tool
#
# Begin!
#
# Message: {input:query}
# Tool: """
# operation_selector = Lamini(
#     "operator", "meta-llama/Llama-2-13b-chat-hf", prompt_template
# )
# prompt_template = """Given:
# {input:query.field} ({input:query.context}): {input:query}
# {input:operation.field} ({input:operation.context}): {input:operation}
# Generate:
# {output:age.field} after "{output:age.field}:"
# {output:age.field}: """
# argument_generator = Lamini(
#     "operator", "meta-llama/Llama-2-13b-chat-hf", prompt_template
# )
# prompt_template = """\
# You are a helpful assistant. You've just been asked to help with a task with the tools:
# {input:operations}
# The user's message: {input:query}
# You decided to use the tool {input:operation} with the arguments {input:args}
# Once you used the tool you got the output {input:output}
# Respond to the user's message
#
# {input:query}
#
# with a final message explaining the actions you took. Do not talk about using tools. Be helpful and inform the user
# of what has happened. If the tool's output includes an error, tell the user that there was an error. Otherwise,
# acknowledge the user's message and tell them what you did: """ vocal_llm = Lamini("vocal",
# "meta-llama/Llama-2-13b-chat-hf", prompt_template) operator = Operator(operation_selector, argument_generator,
# vocal_llm) operator.add_operation(setAge) operator.add_operation(setHeight) output = operator.run("I am 19 years
# old") print("FINAL OUTCOMES: ") print("age:", person_age) print("height:", person_height) print("Final message: ")
# print(output["final_response"])
