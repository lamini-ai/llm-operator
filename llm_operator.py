from llama import Lamini
from typing import Callable, Dict, List, Optional, Any


class Operator:
    def __init__(
        self, operation_selector: Lamini, argument_generator: Lamini, vocal_llm: Lamini
    ) -> None:
        self.operations = {}
        self.argument_generator = argument_generator
        self.operation_selector = operation_selector
        self.vocal_llm = vocal_llm
        self.final_operation = None

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
        input = {"query": query, "operations": str(self.operations)}
        output_type = {
            "operation": "str",
        }
        selected_operation = self.operation_selector(input, output_type)
        return selected_operation

    def select_arguments(
        self,
        query: str,
        operation: str,
    ):
        print("operation: ", operation)
        arguments = self.get_operation_to_run(operation)["arguments"]
        input = {
            "query": query,
            "operation": operation["operation"],
            # "args": arguments,
        }
        output_type = arguments
        generated_arugments = self.argument_generator(
            input,
            output_type,
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
            return self.final_operation(
                input,
                output_type,
            )
        output = self.vocal_llm(
            input,
            output_type,
        )
        return output

    def run(self, query: str):
        # First query the engine with the query string for which operations and arugments to use
        selected_operation = self.select_operations(query)
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
        for name, val in self.operations.items():
            if output["operation"] == name:
                return val
