import os
from llm_operator import Operator

os.environ["LLAMA_ENVIRONMENT"] = "PRODUCTION"


class MotivationOperator(Operator):
    def __init__(self):
        super().__init__()
        self.reminder = None

    def setReminder(self, reminder: str):
        """set a reminder message to do a workout"""
        self.reminder = reminder
        print("setReminder: ", reminder)

    def sendCongrats(self, message: str):
        """send a congratulatory message to the user"""
        print("sendCongrats: ", message)

    def sendMotivationalMessage(self, message: str):
        """send a motivational message to the user"""
        print("sendMotivationalMessage: ", message)

    def __call__(self, mssg):
        self.add_operation(self.setReminder)
        self.add_operation(self.sendCongrats)
        self.add_operation(self.sendMotivationalMessage)
        return self.run(mssg)


if __name__ == '__main__':
    agent = MotivationOperator()
    response = agent("Schedule a workout for 10 pm today.")
    print(response)

    print("FINAL OUTCOMES: ")
    print("age:", agent.reminder)
