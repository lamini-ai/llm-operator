from llm_operator import Operator
import os
# from llama import Lamini

os.environ["LLAMA_ENVIRONMENT"] = "STAGING"

person_age = None


class MotivationOperator:
    reminder_message = None

    def setReminder(reminder: str):
        """set a reminder message to do a workout"""
        print("setReminder: ", reminder)
        global reminder_message
        reminder_message = reminder

    def sendCongrats(message: str):
        """send a congratulatory message to the user"""
        print("sendCongrats: ", message)

    def sendMotivationalMessage(message: str):
        """send a motivational message to the user"""
        print("sendMotivationalMessage: ", message)