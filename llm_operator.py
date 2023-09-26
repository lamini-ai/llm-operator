# from llama import Lamini
from llama import LLMEngine
from llama.prompts.blank_prompt import BlankPrompt
from typing import Callable, Dict, List, Optional, Any
from textwrap import dedent

class Operator:
    def __init__(self) -> None:
        self.operations = {}
        self.prompt = BlankPrompt()
        self.model_name = "meta-llama/Llama-2-7b-chat-hf"
        # self.operation_selector: LLMEngine = self.__generate_operation_name
        # self.argument_generator: LLMEngine = self.__generate_args()
        # self.vocal_llm: LLMEngine = self.__generate_response()
        self.final_operation = None

    @property
    def __generate_operation_name(self):
        prompt_template = """\
        <s>[INST] <<SYS>>
        You are a fitness bot who will infer from user's message what actions should be carried out.
        <</SYS>>
        Use the following format:
        
        'User message': the input message from the user.
        'Tools available': List of tools to choose from.
        'Output': the tool you will use. Write the exact tool name from the choices given.

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
        'User message': {input:query}
        'Tools available': {input:operations}
        [/INST]
        """

        prompt_template = dedent(prompt_template)
        print(prompt_template)
        # operation_selector = Lamini(
        #     "operator", self.model_name, prompt_template
        # )
        operation_selector = LLMEngine(
            id="operator-fw",
            prompt=self.prompt,
            model_name=self.model_name,
        )
        return operation_selector

    def __generate_args(self):
        prompt_template = """<s>[INST] Given:
        {input:query.field} ({input:query.context}): {input:query}
        {input:operation.field} ({input:operation.context}): {input:operation}
        Generate:
        {output:age.field} after "{output:age.field}:"
        {output:age.field}: [INST]"""
        # argument_generator = Lamini(
        #     "operator", self.model_name, prompt_template
        # )
        argument_generator = LLMEngine(
            id="operator-fw",
            prompt=self.prompt,
            model_name=self.model_name,
        )
        return argument_generator

    def __generate_response(self):
        prompt_template = """<s>[INST]
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
        [INST]"""
        # vocal_llm = Lamini("vocal", self.model_name, prompt_template)
        vocal_llm = LLMEngine(
            id="operator-fw",
            prompt=self.prompt,
            model_name=self.model_name,
        )
        return vocal_llm

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

    def select_operations(
            self,
            query: str,
    ):
        operation_descriptions = [
            f"'{name}': '{val['description']}'" for name, val in self.operations.items()
        ]
        input = {"query": query, "operations": str(operation_descriptions)}

        # TODO: check this
        output_type = {
            "operation": "str",
        }
        print("selecting between operations: ", operation_descriptions)
        selected_operation = self.operation_selector(
            input = self.prompt.input(input=input),
            output_type=self.prompt.output,
            stop_tokens = ["\n", ":"]
        )
        return selected_operation

    def select_arguments(
            self,
            query: str,
            operation: str,
    ):
        print("operation: ", operation)
        print("argument_query: ", query)
        arguments = self.get_operation_to_run(operation)["arguments"]
        input = {
            "query": query,
            "operation": operation["operation"],
            # "args": arguments,
        }
        output_type = arguments
        generated_arugments = self.argument_generator(
            input, output_type, stop_tokens=["\n"]
        )
        return generated_arugments

    def get_final_message(
            self,
            query: str,
            selected_operation,
            args: Dict[str, Any],
            operation_output: Any,
    ):
        input = {
            "operations": str(self.operations),
            "query": query,
            "operation": str(selected_operation["operation"]),
            "args": str(args),
            "output": str(operation_output),
        }
        output_type = {
            "final_response": "str",
        }
        if self.final_operation:
            return self.final_operation(input, output_type, stop_tokens=None)
        output = self.vocal_llm(
            input,
            output_type,
        )
        return output

    def run(self, query: str, pass_directly: bool = True):
        # First query the engine with the query string for which operations and arugments to use
        selected_operation = self.select_operations(query)
        if pass_directly:
            action = self.get_operation_to_run(selected_operation)["action"]
            tool_output = action(query)
            generated_arugments = {}
        else:
            # Then query for which arguments to use
            generated_arugments = self.select_arguments(query, selected_operation)
            # Then run the operations
            action = self.get_operation_to_run(selected_operation)["action"]
            tool_output = action(**generated_arugments)
        # Then run the final operation or final llm
        final_message = self.get_final_message(
            query, selected_operation, generated_arugments, tool_output
        )
        return final_message

    def add_final_operation(self, operation: Callable):
        self.final_operation = operation

    def get_operation_to_run(self, output):
        print("get_operation_to_run: ", output)
        for name, val in self.operations.items():
            if output["operation"] == name:
                return val
