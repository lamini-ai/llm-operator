import os
import re
import argparse

from base_operator import Operator
from llama import LlamaV2Runner


os.environ["LLAMA_ENVIRONMENT"] = "PRODUCTION"

class FoodDeliveryOperator(Operator):
    def __init__(self):
        """
        invoke all 'Operator' class methods here.
        Additionally, define any other entities required within any operation.
        """
        super().__init__()
        self.chat_model = LlamaV2Runner()

        # Add operations here
        self.add_operation(self.search)
        self.add_operation(self.order)
        self.add_operation(self.noop)

    def search(self, search_query: str):
        """
        User wants to get an answer about the food delivery app that is available in the FAQ pages of this app. This includes questions about their deliveries, payment, available grocery stores, shoppers, fees, and the app overall.

        Parameters:
        search_query: a query about the order or the app
        """

        # Implement the actual business logic here. Eg: call the search API with this string.
        print("It is indicated that the user wants to search for something.")
        return f"Redirecting to search API with search_query: {search_query}"

    def order(self, item_name: str, quantity: str, unit: str):
        """
        User wants to order items, i.e. buy an item and place it into the cart. This includes any indication of wanting to checkout, or add to or modify the cart. It includes mentioning specific items, or general turn of phrase like 'I want to buy something'.

        Parameters:
        item_name: name of the item that the user wants to order.
        quantity: quantity of the item that the user wants to order.
        unit: unit of the item that the user wants to order like kilograms, pounds, etc.
        """

        # Implement the actual business logic here. Eg: call the order API with this string.
        print("It is indicated that the user wants to invoke cart/order operation.")
        return f"Calling orders API with: item_name={item_name}, quantity={quantity}, unit={unit}"

    def noop(self, message: str):
        """
        User didn't specify a tool, i.e. they didn't say they wanted to search or order. The ask is totally irrelevant to the delivery service app.

        Parameters:
        message: a message/query not related to the app.
        """

        # Implement the actual business logic here. Eg: save this data in 'miscellaneous data' for user search analysis.
        print("It is indicated that this is a general query. So redirecting to a chat LLM.")
        model_response = self.chat_model(message, system_prompt="answer in 3 sentences maximum.")
        clean_response = re.sub(r'\.{2,}', '.', model_response)
        return f"Calling general query LLM...\nuser query= {message} \n\noutput=\n{clean_response}"


def train(operator_save_path, training_data=None):
    """Trains the Operator."""
    operator = FoodDeliveryOperator()
    operator.train(training_data, operator_save_path)
    print('Done training!')

def inference(queries, operator_save_path):
    operator = FoodDeliveryOperator().load(operator_save_path)
    
    for query in queries:
        print(f"\n\nQuery: {query}")
        response = operator(query)
        print(response)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--operator_save_path",
        type=str,
        help="Path to save the operator / use the saved operator.",
        default="models/FoodDeliveryOperator/",
    )

    parser.add_argument(
        "--training_data",
        type=str,
        help="Path to dataset (CSV) to train on. Optional.",
        default="data/food_delivery.csv",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model.",
        default=False,
    )

    parser.add_argument(
        "--query",
        type=str,
        nargs="+",
        action="extend",
        help="Queries to run",
        default=[],
    )

    args = parser.parse_args()

    if args.operator_save_path[-1] != "/":
        args.operator_save_path += "/"

    if args.train:
        train(args.operator_save_path, args.training_data)
    
    default_queries = ["I want to order 2 gallons of milk.", "What are the benefits of upgrading my membership?", "Are there any exercises I can do to lose weight?"]
    queries = args.query if args.query else default_queries
    inference(queries, args.operator_save_path)

if __name__ == '__main__':
    main()



