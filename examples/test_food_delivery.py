import os

from llm_operator import Operator

os.environ["LLAMA_ENVIRONMENT"] = "PRODUCTION"


class FoodDeliveryOperator(Operator):
    def search(self, search_query: str):
        """
        User wants to get an answer about the food delivery app that is available in the FAQ pages of this app. This includes questions about their deliveries, payment, available grocery stores, shoppers, fees, and the app overall.

        Parameters:
        search_query: a query about the order or the app
        """

        # Implement the actual business logic here. Eg: call the search API with this string.
        return f"search_query: {search_query}"

    def order(self, order_query: str):
        """
        User wants to order items, i.e. buy an item and place it into the cart. This includes any indication of wanting to checkout, or add to or modify the cart. It includes mentioning specific items, or general turn of phrase like 'I want to buy something'.

        Parameters:
        order_query: query related ot the order
        """

        # Implement the actual business logic here. Eg: call the order API with this string.
        return "order:" + order_query

    def noop(self, message: str):
        """
        User didn't specify a tool, i.e. they didn't say they wanted to search or order. The ask is totally irrelevant to the delivery service app.

        Parameters:
        message: a random message not related to the app.
        """

        # Implement the actual business logic here. Eg: save this data in 'junk data' for user search analysis.
        return "noop:" + message

    def __call__(self, mssg, train = False, training_data_path=None):
        self.add_operation(self.search)
        self.add_operation(self.order)
        self.add_operation(self.noop)
        if train:
            agent.train_router(training_data_path)
        return self.run(mssg)


if __name__ == '__main__':
    agent = FoodDeliveryOperator(router_path="examples/models/clf/FoodDeliveryOperator")
    query = "I want 5l of milk."
    response = agent(query, False)
    print(response)

