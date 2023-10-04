import os
import re
import argparse
import random

from base_operator import Operator
from llama import LlamaV2Runner


os.environ["LLAMA_ENVIRONMENT"] = "PRODUCTION"

class CustomerSupportOperator(Operator):
    def __init__(self):
        super().__init__()
        self.chat_model = LlamaV2Runner()

        # Add operations here
        self.add_operation(self.create_ticket)
        self.add_operation(self.close_ticket)
        self.add_operation(self.escalate)
        self.add_operation(self.gather_info)

    def create_ticket(self, ticket_category: str):
        """
        User has provided enough information in the chat to create a support ticket about their issue. This includes technical support issues relating to the app, billing, and user's account.
        
        Parameters:
        ticket_category: the category of the user's issue. Eg: app, billing, account.
        """

        # Implement the actual business logic here. Eg: call the create ticket API with this string.
        return f"Redirecting to create ticket API with description={ticket_category}"

    def close_ticket(self, is_closed: str):
        """
        User issue is resolved. Close the ticket.

        Parameters:
        is_closed: 'yes' or 'no'. Indicates whether the ticket is closed.
        """

        # Implement the actual business logic here. Eg: call the close ticket API with this string.
        if is_closed == "yes":
            return f"The user's issue is resolved. The ticket is closed."
        else:
            return f"The user's issue is not resolved. The ticket is still open."

    def escalate(self, severity_level: str):
        """
        User issue is not resolved after multiple tries. Escalate the ticket.

        Parameters:
        severity_level: high, medium, low. Indicates the severity of the issue.
        """

        # Implement the actual business logic here. Eg: call the escalate ticket API with this string.
        return f"The user's issue is not resolved. The ticket is escalated with the level={severity_level}"

    def gather_info(self, message: str):
        """
        User continues to provide information about their issue. It is helpful to continue asking the user for more information until the issue is resolved.

        Parameters:
        message: the user's last message.
        """

        # Implement the actual business logic here. Eg: save this data in 'miscellaneous data' for user search analysis.
        model_response = self.chat_model(message, system_prompt="Your job is to get more details on the user's issue. Answer the user's questions, or ask the user for more details. Use 1 sentence.")
        clean_response = re.sub(r"(\.|\?){2,}", r"\1", model_response)
        return f"Continuing the conversation with the user. Calling a chat LLM... \nresponse=\n{clean_response}"


def train(operator_save_path, training_data=None):
    """Trains the Operator."""
    operator = CustomerSupportOperator()
    operator.train(operator_save_path, training_data)
    print('Done training!')

def inference(queries, operator_save_path):
    operator = CustomerSupportOperator().load(operator_save_path)
    for query in queries:
        print(f"\n\nUser message: {query}")
        response = operator(query)
        print(response)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--operator_save_path",
        type=str,
        help="Path to save the operator / use the saved operator.",
        default="models/CustomerSupportOperator/",
    )

    parser.add_argument(
        "--training_data",
        type=str,
        help="Path to dataset (CSV) to train on. Optional.",
        default="data/customer_support.csv",
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

    if args.train:
        train(args.operator_save_path, args.training_data)
    
    default_queries = [
        "can't login",
        "great, thanks!",
        "can I talk to your manager?",
        "hi there I'd like to understand my bill",
    ]
    queries = args.query if args.query else default_queries
    inference(queries, args.operator_save_path)

if __name__ == '__main__':
    main()



