# from llama import Lamini
from llama import LLMEngine
from llama.prompts.blank_prompt import BlankPrompt
from typing import Callable, Dict, List, Optional, Any
from textwrap import dedent
import re
import ast

'''
DEPRECATED
'''
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

    #     Example:
    #     Given:
    #     'User message': 'I am Nathan and I am 20 years old.'
    #     'Tool chosen': [
    #         'setAge': age
    #     of
    #     the
    #     user in years.
    #     ]
    #     'Output types': [{{'name': 'age', 'type': 'int', description: 'age of the user'}}]
    #
    # Generate:
    # 'Result': {{'age': '20'}}
    def __generate_args(self):
        prompt_template = """\
        <s>[INST] For the given operation, find out what arguments to call the tool with.
         For the following input format:
        'User message': the input message from the user.
        'Tool chosen': tool chosen and its function.
        'Output types': the tool requires some arguments to be passed to it. This includes a list of dictionaries containing 'name' which is the name of the argument to be passed to the tool, 'type' which is the type of the argument, 'description' which is the meaning of the argument.
        
        Output should be of format:
        'Output': a dictionary containing the arguments and values as parsed from user message.
        
        Given:
            'User message': {query}
            'Tool chosen': {operation}
            'Output types': {args}
        generate the 'Output' only. Do not explain the logic.
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

    def get_func_args(self, op):
        args = []
        for key, value in op.__annotations__.items():
            pattern = fr"{key}:\s(.*?)\.?\n"
            match = re.search(pattern, op.__doc__, re.DOTALL)
            if match:
                param_description = match.group(1).strip()
            else:
                param_description= None
            args.append({"name": key, "type": str(value), "description": param_description})
        return args

    def add_operation(
            self,
            operation,
            description: Optional[str] = None,
    ):
        name = operation.__name__
        description = description or operation.__doc__
        arguments = self.get_func_args(operation)
        self.operations[name] = {
            "action": operation,
            "description": description,
            "arguments": str(arguments),
        }


    def parse_op_output(self, response):
        output_match = re.search(r"'Output':\s*'([^']+)'", response)

        if output_match:
            output_value = output_match.group(1)
            return output_value
        else:
            print("\nPattern 'Output' not found in the input text.")

    def parse_extractor_output(self, response):
        output_match = re.search(r"'Output':\s*({.*?})", response)

        if output_match:
            output_value = output_match.group(1)
            return output_value
        else:
            print("\nPattern 'Output' not found in the input text.")

    def select_operations(
            self,
            query: str,
    ):
        operation_descriptions = []
        for name, val in self.operations.items():
            desc = val['description'].split("\n")[1].strip()
            operation_descriptions.append(f"{name}: {desc}")
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
        selected_operation = self.parse_op_output(model_response.output)
        print("selected_operation:", selected_operation)
        return selected_operation

    # def get_func_args(self, op):
    #     # param_list = []
    #     # pattern = r"(\w+)\s+\((\w+)\):\s*(.*)"
    #     #
    #     # print("ayu:", self.operations[op])
    #     # print("ayu2:", self.operations[op].__annotations__)
    #     # docstring = self.operations[op].description
    #     # matches = re.findall(pattern, docstring)
    #     # for match in matches:
    #     #     param_name, param_type, param_description = match
    #     #     param_list.append(
    #     #         f"{{'field': {param_name}, 'type': {param_type}, 'description': {param_description} }}"
    #     #     )
    #     param_list = []
    #     operation = self.get_operation_to_run(op)
    #     # print("ayu:", operation)
    #     function = getattr(self, op, None)
    #     if function is not None and inspect.ismethod(function):
    #         signature = inspect.signature(function)
    #         parameters = signature.parameters
    #         print(f"signature: {signature}, parameters: {parameters}")
    #         for parameter_name, parameter in parameters.items():
    #             parameter_type = parameter.annotation
    #             parameter_desc = inspect.getdoc(parameter)
    #             print(f"AyuParameter: {parameter_name}, Type: {parameter_type},  'description': {parameter_desc}")
    #
    #             #  'Output types': [{{'field': 'age', 'type': 'int', description: 'age of the user'}}]
    #             param_list.append(f"{{'field': {parameter_name}, 'type': {parameter_type}, 'description': {parameter_desc} }}")
    #     return param_list

    def select_arguments(
            self,
            query: str,
            operation: str,
    ):
        print("operation: ", operation)
        print("argument_query: ", query)

        arguments = self.get_operation_to_run(operation)['arguments']
        input = {
            "query": query,
            "operation": operation,
            "args": arguments
        }
        # output_type = arguments
        # generated_arugments = self.argument_generator(
        #     input, output_type, stop_tokens=["\n"]
        # )
        prompt_template = self.__generate_args()
        prompt_str = prompt_template.format_map(input)
        print("select_arguments:", prompt_str)
        model_response = self.model(
            input=self.prompt.input(input=prompt_str),
            output_type=self.prompt.output,
            stop_tokens=["\n", ":"]
        )
        # TODO: remove when using jsonformer
        print("\n\nselect_arguments resp:", model_response.output)
        generated_arguments = self.parse_extractor_output(model_response.output)
        print("generated_arguments:", generated_arguments)
        return ast.literal_eval(generated_arguments)

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
        for name, val in self.operations.items():
            if output == name:
                return val
