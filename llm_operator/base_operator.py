import re
import os
from textwrap import dedent
from typing import Optional

from llama import Lamini
from llama.prompts.blank_prompt import BlankPrompt
from llm_routing_agent import LLMRoutingAgent


class Operator:
    def __init__(self) -> None:
        self.operations = {}
        self.prompt = BlankPrompt()
        self.model_load_path = None
        self.model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.routing_threshold = 0.3

    def load(self, path):
        '''
        Load the routing operator from the given path.
        '''
        self.model_load_path = path
        if not os.path.exists(self.model_load_path):
            raise Exception("Operator path does not exist. Please train your operator first or check the path passed.")
        for operator_name in self.operations:
            model_path = os.path.join(self.model_load_path, operator_name + ".pkl")
            if not os.path.exists(model_path):
                raise Exception(
                    f"Model for operation {operator_name} not found. Please train your operator for this operation as well.")
        return self

    def __generate_args_prompt(self):
        prompt_template = """\
        <s>[INST] <<SYS>> For the given operation, find out the values of the arguments to call the tool with. For the following input format:
        'User message': the input message from the user.
        'Tool chosen': tool chosen and its function.
        'Arguments list': the list of arguments required for the tool. This includes argument name, type and description.

        Output format:
        'Output': a dictionary of argument names and values.
        <</SYS>>

        Given:
        'User message': {input:query}
        'Tool chosen': {input:operation}
        'Arguments list': {input:args}
        generate the 'Output' only. Do not explain the logic.
        [/INST] """

        prompt_template = dedent(prompt_template)
        model = Lamini(
            "operator", self.model_name, prompt_template
        )
        return model

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
            "arguments": arguments,
        }

    def select_operations(self, query, classes_dict):
        '''
        selects which tools to use
        '''
        router = LLMRoutingAgent(self.model_load_path, self.routing_threshold)
        predicted_ops = router.predict([query], classes_dict)
        return predicted_ops[0]

    def select_arguments(
            self,
            query: str,
            operation: str,
    ):
        '''
        Predicts and parses the arguments required to call the tool.
        '''
        arguments = self.__get_operation_to_run(operation)['arguments']
        output_type = {}
        for arg in arguments:
            output_type[arg['name']] = arg['type']
        input = {
            "query": query,
            "operation": operation,
            "args": str(arguments)
        }
        model = self.__generate_args_prompt()
        model_response = model(
            input,
            output_type,
            stop_tokens=["</s>"]
        )
        return model_response

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

        self.model_load_path = router_save_path
        # if os.path.exists(self.model_load_path):
        #     print("Operator already trained. Loading from saved path.")
        #     self.load(router_save_path)
        #     return
        router = LLMRoutingAgent(self.model_load_path, self.routing_threshold)
        classes_dict = self.__get_classes_dict()
        router.fit(classes_dict, training_file)

    def run(self, query: str):
        '''
        Gets the routing agent to decide which tool to use.
        Next, this agent finds out the value of the arguments required to call that tool.
        That tool is then called. Tool output is returned.
        '''
        if not self.model_load_path:
            raise Exception("Operator not loaded.")

        classes_dict = self.__get_classes_dict()
        selected_operations = self.select_operations(query, classes_dict)
        print(f"\nselected operations: {selected_operations}")
        t = []
        for op in selected_operations:
            generated_arguments = self.select_arguments(query, op)
            print(f"inferred arguments for {op}: {generated_arguments}")
            action = self.__get_operation_to_run(op)["action"]
            tool_output = action(**generated_arguments)
            t.append(tool_output)
        return t

    def __call__(self, query: str):
        return self.run(query)
