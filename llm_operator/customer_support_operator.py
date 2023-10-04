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

    def create_ticket(self, ticket_summary: str):
        """
        User has provided enough information in the chat to create a support ticket about their issue. This includes technical support issues relating to the app, billing, and user's account.
        
        Parameters:
        ticket_summary: a one-sentence summary of the issue that the user is facing.
        """

        # Implement the actual business logic here. Eg: call the create ticket API with this string.
        print("Creating a ticket about the user's issue and categorizing the issue.")
        return f"Redirecting to create ticket API with the description={ticket_summary}"

    def close_ticket(self, resolution_reason: str):
        """
        User issue is resolved. Close the ticket.

        Parameters:
        resolution_reason: summary of how the user's issue was resolved.
        """

        # Implement the actual business logic here. Eg: call the close ticket API with this string.
        print("It is indicated that the user's issue is resolved. Closing the ticket")
        return f"The ticket is closed."

    def escalate(self, severity_level: str):
        """
        User issue is not resolved after multiple tries. Escalate the ticket.

        Parameters:
        severity_level: high, medium, low. Indicates the severity of the issue.
        """

        # Implement the actual business logic here. Eg: call the escalate ticket API with this string.
        print("It is indicated that the user's issue is not resolved. Escalating the ticket.")
        return f"The ticket is escalated with the level={severity_level}"

    def gather_info(self, chat_history: str, message: str):
        """
        User continues to provide information about their issue. It is helpful to continue asking the user for more information until the issue is resolved.

        Parameters:
        chat_history: a summary of the chat history so far.
        message: the user's last message.
        """

        # Implement the actual business logic here. Eg: save this data in 'miscellaneous data' for user search analysis.
        print(f"chat_history={chat_history}")
        print("Continuing the conversation with the user.")
        model_response = self.chat_model(message, system_prompt="Your job is to get more details on the user's issue. Answer the user's questions, or ask the user for more details. Use 1 sentence.")
        clean_response = re.sub(r"(\.|\?){2,}", r"\1", model_response)
        return f"Calling a chat LLM...\nmessage={message}\n\noutput=\n{clean_response}"


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



