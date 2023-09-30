import ast
import re
from textwrap import dedent
from typing import Optional
from llama import LLMEngine
from llama.prompts.blank_prompt import BlankPrompt
from routing_operator import RoutingOperator
import os

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
        self.router=None
        self.model_load_path = None

    def load(self, router_path):
        '''
        Load the routing operator from the given path.
        '''
        if not os.path.exists(router_path):
            raise Exception("Router path does not exist.")
        if router_path and not router_path.endswith(".pkl"):
            raise Exception("Model pickle file not detected.")
        self.model_load_path = router_path
        self.router = RoutingOperator(self.model_load_path)
        return self

    def __generate_args(self):
        prompt_template = """\
        <s>[INST] For the given operation, find out what arguments to call the tool with.
         For the following input format:
        'User message': the input message from the user.
        'Tool chosen': tool chosen and its function.
        'Output types': the tool requires some arguments to be passed to it. This includes a list of dictionaries containing 'name' which is the name of the argument to be passed to the tool, 'type' which is the type of the argument, 'description' which is the meaning of the argument.

        Output should be of format:
        'Output': a json containing the argument names and values as parsed from user message.

        Given:
            'User message': {query}
            'Tool chosen': {operation}
            'Output types': {args}
        generate the 'Output' only. Do not explain the logic.
        [/INST]
        """
        prompt_template = dedent(prompt_template)
        return prompt_template

    def get_func_args(self, op):
        '''
        currently getting the docstring or function annotation doesn't give parameter wise description. so we try to find each function parameter and its specific description.
        '''
        args = []
        for key, value in op.__annotations__.items():
            pattern = fr"{key}:\s(.*?)\.?\n"
            match = re.search(pattern, op.__doc__, re.DOTALL)
            if match:
                param_description = match.group(1).strip()
            else:
                param_description = None
            args.append({"name": key, "type": "string", "description": param_description})
        return args

    def add_operation(
            self,
            operation,
            description: Optional[str] = None,
    ):
        '''
        Add tools to the agent. Each tool has tool name, description and arguments required.
        '''
        name = operation.__name__
        description = description or operation.__doc__.split("\n")[1].strip()
        arguments = self.get_func_args(operation)
        self.operations[name] = {
            "action": operation,
            "description": description,
            "arguments": str(arguments),
        }

    def select_operations(self, query):
        '''
        selects which tool to use
        '''
        # TODO: predict multiple operations
        predicted_cls, prob = self.router.predict([query])
        return predicted_cls[0]

    def select_arguments(
            self,
            query: str,
            operation: str,
    ):
        '''
        Predicts and parses the arguments required to call the tool.
        '''
        arguments = self.__get_operation_to_run(operation)['arguments']
        input = {
            "query": query,
            "operation": operation,
            "args": arguments
        }
        prompt_template = self.__generate_args()
        prompt_str = prompt_template.format_map(input)
        model_response = self.model(
            input=self.prompt.input(input=prompt_str),
            output_type=self.prompt.output
        )
        # TODO: remove when using jsonformer
        generated_arguments = self.__parse_argument_output(model_response.output)
        return ast.literal_eval(generated_arguments)

    def __parse_argument_output(self, response):
        '''
        Parse the exact argument output.
        '''
        # output_match = re.search(r"'Output':\s*({.*?})", response)
        output_match = re.search(r"'Output':\s*({[^}]+})", response)

        if output_match:
            output_value = output_match.group(1)
            return output_value
        else:
            print("\nPattern 'Output' not found in the input text.")

    def __get_operation_to_run(self, output):
        '''
        Get the tool callback from the name of the tool.
        '''
        for name, val in self.operations.items():
            if output == name:
                return val

    def __get_classes_dict(self):
        '''
        get tool name and description list
        '''
        cls = {}
        for name, val in self.operations.items():
            cls[name] = val["description"]
        return cls

    def train(self, training_file, router_save_path):
        '''
        Train the routing agent to decide which tool to use.
        '''
        if not os.path.exists(router_save_path):
            os.makedirs(router_save_path)
        if training_file and not os.path.exists(training_file):
            print("Training file does not exist. Continuing without it.")

        self.model_load_path = router_save_path + "router.pkl"
        self.router = RoutingOperator(self.model_load_path)
        classes_dict = self.__get_classes_dict()
        self.router.fit(classes_dict, training_file)
        self.router.save(self.model_load_path)

    def run(self, query: str):
        '''
        Gets the routing agent to decide which tool to use.
        Next, this agent finds out the value of the arguments required to call that tool.
        That tool is then called. Tool output is returned.
        '''
        if not self.model_load_path:
            raise Exception("Router not loaded.")
        selected_operation = self.select_operations(query)
        print(f"selected_operation: {selected_operation}")
        generated_arugments = self.select_arguments(query, selected_operation)
        print(f"inferred_arugments: {generated_arugments}")
        action = self.__get_operation_to_run(selected_operation)["action"]
        tool_output = action(**generated_arugments)
        return tool_output
