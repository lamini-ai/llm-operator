import os

from llm_operator import Operator
from llama import LlamaV2Runner
os.environ["LLAMA_ENVIRONMENT"] = "PRODUCTION"
import re

class FoodDeliveryOperator(Operator):
    def __init__(self):
        super().__init__()
        self.chat_model = LlamaV2Runner()

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
        print("It is indicated that the user wants to place an order.")
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

    def __call__(self, mssg):
        return self.run(mssg)


if __name__ == '__main__':
    # train and  inference
    # #optional training file path
    # training_file = None
    # router_save_path = "examples/models/clf/FoodDeliveryOperator/"
    # foodOperator = FoodDeliveryOperator()
    # foodOperator.add_operation(foodOperator.search)
    # foodOperator.add_operation(foodOperator.order)
    # foodOperator.add_operation(foodOperator.noop)
    # foodOperator.train(training_file, router_save_path)
    # query = "I want 10l of milk."
    # response = foodOperator(query)

    # inference
    router_save_path = "examples/models/clf/FoodDeliveryOperator/router.pkl"
    foodOperator = FoodDeliveryOperator().load(router_save_path)
    foodOperator.add_operation(foodOperator.search)
    foodOperator.add_operation(foodOperator.order)
    foodOperator.add_operation(foodOperator.noop)
    query1 = "I want 10l of milk."
    print(f"\n\nQuery: {query1}")
    response1 = foodOperator(query1)
    print(response1)
    query2 = "What are the benefits of upgrading my membership?"
    print(f"\n\nQuery2: {query2}")
    response2 = foodOperator(query2)
    print(response2)
    query3 = "Are there any exercises I can do to lose weight?"
    print(f"\n\nQuery2: {query3}")
    response3= foodOperator(query3)
    print(response3)


