import os
from pydantic import BaseModel
from llama import LLMEngine
from llama import Lamini
from typing import Callable, Dict, List, Optional, Any
import inspect
from llama.types.type import Type


class Operator:
    def __init__(
        self, operation_selector: Lamini, argument_generator: Lamini, vocal_llm: Lamini
    ) -> None:
        self.operations = {}
        self.argument_generator = argument_generator
        self.operation_selector = operation_selector
        self.vocal_llm = vocal_llm
        self.final_callback = None

    def add_operation(
        self,
        callback,  # Callable[[Dict[str, Any]], MultiOperationOutput],
        description: Optional[str] = None,
    ):
        name = callback.__name__
        description = description or callback.__doc__
        # arguments = str(inspect.signature(callback))
        arguments = {key: str(value) for key, value in callback.__annotations__.items()}
        self.operations[name] = {
            "action": callback,
            "description": description,
            "arguments": arguments,
        }

    def run(self, query: str):
        # First query the engine with the query string for which operations and arugments to use
        input = {"query": query, "operations": str(self.operations)}
        output_type = {
            "operation": "str",
        }
        selected_operation = self.operation_selector(input, output_type)
        # print("output1:", selected_operation)

        # Then query for which arguments to use
        operation = self.get_callback_to_run(selected_operation)
        arguments = operation["arguments"]
        callback = operation["action"]
        input = {
            "query": query,
            "operation": selected_operation["operation"],
            # "args": arguments,
        }
        output_type = arguments
        generated_arugments = self.argument_generator(
            input,
            output_type,
        )
        # print("output2:", generated_arugments)

        # Then run the callbacks
        tool_output = callback(**generated_arugments)

        # Then run the final callback or final llm

        if self.final_callback:
            return self.final_callback(tool_output)

        input = {
            "operations": str(self.operations),
            "query": query,
            "operation": str(selected_operation["operation"]),
            "args": str(generated_arugments),
            "output": str(tool_output),
        }
        output_type = {
            "final_response": "str",
        }
        # print("final vocal input:", input)
        output = self.vocal_llm(
            input,
            output_type,
        )
        return output

    def add_final_callback(self, callback: Callable):
        self.final_callback = callback

    def get_callback_to_run(self, output):
        for name, val in self.operations.items():
            if output["operation"] == name:
                return val


person_age = None


def setAge(age: int):
    """set the age of a person"""
    global person_age
    person_age = age


person_height = None


def setHeight(height: int):
    """set the height of a person in inches"""
    global person_height
    person_height = height


# def callOnboardingAgent(message: str):
#     """Use onboarding agent"""
#     onboardingOperator = Operator()
#     onboardingOperator.add_operation(setAge)
#     output = onboardingOperator.query(message)
#     return output

# reminder_message = ""

# def setReminder(reminder: str):
#     """set a reminder"""
#     global reminder_message
#     reminder_message = reminder

# def callMotivationAgent(message: str):
#     """Use motivation agent"""
#     motivationAgent = Operator()
#     motivationAgent.add_operation(setReminder)
#     motivationAgent.add_operation(sendCongrats)
#     output = motivationAgent.query(message)
#     return output

# routerOperator = Operator()
# routerOperator.add_operation(callOnboardingAgent)
# routerOperator.add_operation(callMotivationAgent)

# response = routerOperator.query("I am 19 years old")
# print(response)


if __name__ == "__main__":
    os.environ["LLAMA_ENVIRONMENT"] = "STAGING"
    prompt_template = """\
Respond to the message using function calls. You have access to the following tools:

{input:operations}

Use the following format:

Message: the input message you must use
Tool: the tool you will use. Only write the name of the tool

Begin!

Message: {input:query}
Tool: """
    operation_selector = Lamini(
        "operator", "meta-llama/Llama-2-13b-chat-hf", prompt_template
    )
    prompt_template = """Given:
{input:query.field} ({input:query.context}): {input:query}
{input:operation.field} ({input:operation.context}): {input:operation}
Generate:
{output:age.field} after "{output:age.field}:"
{output:age.field}: """
    argument_generator = Lamini(
        "operator", "meta-llama/Llama-2-13b-chat-hf", prompt_template
    )
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
    operator = Operator(operation_selector, argument_generator, vocal_llm)
    operator.add_operation(setAge)
    operator.add_operation(setHeight)
    output = operator.run("I am 19 years old")
    print("FINAL OUTCOMES: ")
    print("age:", person_age)
    print("height:", person_height)
    print("Final message: ")
    print(output["final_response"])
