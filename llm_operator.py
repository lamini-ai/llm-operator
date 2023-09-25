# from llama import Lamini
from typing import Callable, Dict, List, Optional, Any
import re

class Operation:
    def __init__(self, name):
        self.name = name

    def run(self, terminal_task = False):
        prompt = f'''For the given input, perform the task. {self.description}'''
        return self.llm("operator", "meta-llama/Llama-2-13b-chat-hf", prompt)


class Operator:
    def __init__(self, name):
        self.name = ""
        self.function = eval(name)
        self.prompt = ""
        self.llm = None

    def __inject(self, original_string, pattern, replacement_string):
        return re.sub(pattern, replacement_string, original_string)

    def add_operations(self, cls, ops: List[Operation]):
        if len(ops) == 0:
            raise Exception("Atleast 1 operation needs to be passed")

        tools_str = ""
        for op in ops:
            if hasattr(cls, op.name) and callable(getattr(cls, op.name)):
                result = getattr(cls, op.name)()
                print(result)
            else:
                print(f"Function '{op}' not found in Operator.")

        # for op in ops:
        #     tools_str += f"{op.name}: {op.description},\n"
        # self.__inject(self.prompt, "input:tools", tools_str)

    def run(self):
        if len(self.operations) == 0:
            raise Exception("Atleast 1 operation needs to be passed")




