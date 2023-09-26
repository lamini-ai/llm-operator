# from llama import Lamini
from llama import LLMEngine
from llama.prompts.blank_prompt import BlankPrompt
from typing import Callable, Dict, List, Optional, Any
from textwrap import dedent
import re


class Operator:
    def __init__(self) -> None:
        self.operations = {}
        self.prompt = BlankPrompt()
        self.model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.model = LLMEngine(
            id="operator-fw",
            prompt=self.prompt,
            model_name=self.model_name,
        )
        # self.operation_selector: LLMEngine = self.__generate_operation_name
        # self.argument_generator: LLMEngine = self.__generate_args()
        # self.vocal_llm: LLMEngine = self.__generate_response()
        self.final_operation = None

    def __generate_operation_name(self):
        prompt_template = """\
        <s>[INST] <<SYS>>
        You are a fitness bot who will infer from user's message what actions should be carried out.
        <</SYS>>
        Use the following format:
        
        'User message': the input message from the user.
        'Tools available': List of tools to choose from.
        'Output': the tool you will use. Write only the exact tool name from the choices given.

        Examples:
        'User message': 'I feel a striking pain in my chest.'
        'Tools available': [
        'Book an appointment': to book an appointment,
        'Cancel an appointment': to cancel an appointment, 
        'Book an emergency appointment': to book an emergency appointment, 
        'Financials and payment': patient payment related question, 
        'Other': any other questions
        ]
        'Output': ['Book an emergency appointment appointment']

        Now for the following 'User message', share the 'Output'.
        'User message': {query}
        'Tools available': {operations}
        [/INST]
        """

        prompt_template = dedent(prompt_template)
        return prompt_template

    def __generate_args(self):
        prompt_template = """\
        <s>[INST] For the given operation, find out what arguments to call the tool with.
         Examples:
        'User message': 'I am 15 years old and weigh 100 lbs.'
        'Tool chosen': [
        'setAge': age of the user in years.
        ]
        'Output': {'age': '15'}

        Now for the following 'User message', share the 'Output'.
        'User message': {query}
        'Tool chosen': {operation}
        [/INST]
        """
        # argument_generator = Lamini(
        #     "operator", self.model_name, prompt_template
        # )
        prompt_template = dedent(prompt_template)
        return prompt_template

    # def __generate_response(self):
    #     prompt_template = """\
    #     <s>[INST]
    #     You are a helpful assistant. You've just been asked to help with a task with the tools:
    #     {operations}
    #
    #     for the  user's message:
    #     {query}
    #     You decided to use the tool {operation} with the arguments {args}
    #     Once you used the tool you got the output- '{output}'
    #     Respond to the user with the output of that tool exactly.
    #     [INST]"""
    #     # vocal_llm = Lamini("vocal", self.model_name, prompt_template)
    #     prompt_template = dedent(prompt_template)
    #     return prompt_template

    def add_operation(
            self,
            operation,
            description: Optional[str] = None,
    ):
        name = operation.__name__
        description = description or operation.__doc__
        arguments = {
            key: str(value) for key, value in operation.__annotations__.items()
        }
        self.operations[name] = {
            "action": operation,
            "description": description,
            "arguments": arguments,
        }

    def parse_output(self, response):
        output_match = re.search(r"'Output':\s*\"([^\"]+)\"", response)

        if output_match:
            output_value = output_match.group(1)
            return output_value.strip()
        else:
            raise Exception("Pattern 'Output' not found in the input text.")

    def select_operations(
            self,
            query: str,
    ):
        operation_descriptions = [
            f"'{name}': '{val['description']}'" for name, val in self.operations.items()
        ]
        print("selecting between operations: ", operation_descriptions)
        input = {"query": query, "operations": str(operation_descriptions)}

        prompt_template = self.__generate_operation_name()
        prompt_str = prompt_template.format_map(input)
        print("\n", prompt_str)

        # TODO: check this
        output_type = {
            "operation": "str",
        }

        model_response = self.model(
            input=self.prompt.input(input=prompt_str),
            output_type=self.prompt.output,
            stop_tokens=["\n", ":"]
        )
        # TODO: remove when using jsonformer
        print("\n\nresp:", model_response.output)
        selected_operation = self.parse_output(model_response.output)
        print("selected_operation:", selected_operation)
        return selected_operation

    def select_arguments(
            self,
            query: str,
            operation: str,
    ):
        print("operation: ", operation)
        print("argument_query: ", query)
        arguments = self.get_operation_to_run(operation)
        input = {
            "query": query,
            "operation": operation,
        }
        # output_type = arguments
        # generated_arugments = self.argument_generator(
        #     input, output_type, stop_tokens=["\n"]
        # )
        prompt_template = self.__generate_args()
        print(prompt_template)
        prompt_str = prompt_template.format_map(input)
        model_response = self.model(
            input=self.prompt.input(input=prompt_str),
            output_type=self.prompt.output,
            stop_tokens=["\n", ":"]
        )
        # TODO: remove when using jsonformer
        print("\n\nresp:", model_response.output)
        generated_arguments = self.parse_output(model_response.output)
        print("generated_arguments:", generated_arguments)
        return generated_arguments

    # def get_final_message(
    #         self,
    #         query: str,
    #         selected_operation,
    #         args: Dict[str, Any],
    #         operation_output: Any,
    # ):
    #     operation_descriptions = [
    #         f"'{name}': '{val['description']}'" for name, val in self.operations.items()
    #     ]
    #     input = {
    #         "operations": str(operation_descriptions),
    #         "query": query,
    #         "operation": str(selected_operation),
    #         "args": str(args),
    #         "output": str(operation_output),
    #     }
    #     # output_type = {
    #     #     "final_response": "str",
    #     # }
    #     # if self.final_operation:
    #     #     return self.final_operation(input, output_type, stop_tokens=None)
    #     # output = self.vocal_llm(
    #     #     input,
    #     #     output_type,
    #     # )
    #     output_type = {
    #         "final_response": "str",
    #     }
    #     if self.final_operation:
    #         return self.final_operation(input, output_type, stop_tokens=None)
    #
    #     prompt_template = self.__generate_response()
    #
    #     prompt_str = prompt_template.format_map(input)
    #     print("\n", prompt_str)
    #     model_response = self.model(
    #         input=self.prompt.input(input=prompt_str),
    #         output_type=self.prompt.output,
    #         stop_tokens=["\n", ":"]
    #     )
    #     # TODO: remove when using jsonformer
    #     print("\n\nresp:", model_response.output)
    #     return model_response.output

    def run(self, query: str, pass_directly: bool = True):
        # First query the engine with the query string for which operations and arugments to use
        selected_operation = self.select_operations(query)
        if pass_directly:
            action = self.get_operation_to_run(selected_operation)["action"]
            tool_output = action(query)
        else:
            # Then query for which arguments to use
            generated_arugments = self.select_arguments(query, selected_operation)
            # Then run the operations
            action = self.get_operation_to_run(selected_operation)["action"]
            tool_output = action(**generated_arugments)
        # Then run the final operation or final llm
        # final_message = self.get_final_message(
        #     query, selected_operation, generated_arugments, tool_output
        # )
        return tool_output

    def add_final_operation(self, operation: Callable):
        self.final_operation = operation

    def get_operation_to_run(self, output):
        print("get_operation_to_run: ", output)
        for name, val in self.operations.items():
            if output == name:
                return val
