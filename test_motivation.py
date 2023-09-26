import os
from llm_operator import Operator

os.environ["LLAMA_ENVIRONMENT"] = "STAGING"


class MotivationOperator(Operator):
    def __init__(self):
        super().__init__()

    def setReminder(self, workout_name: str, workout_time: str):
        """set a reminder message to the user to do workout by the name 'workout_name' at 'workout_time'. 'workout_time' is in datetime format."""
        print("setReminder: ")
        return f"Reminder has been set. Workout: {workout_name}, Time: {workout_time}"

    def sendCongrats(self, message: str):
        """send a congratulatory message 'message' to the user"""
        print("sendCongrats: ", message)

    def sendMotivationalMessage(self, message: str):
        """send a motivational message to the user"""
        print("sendMotivationalMessage: ", message)

    def __call__(self, mssg):
        self.add_operation(self.setReminder)
        self.add_operation(self.sendCongrats)
        self.add_operation(self.sendMotivationalMessage)
        return self.run(mssg, False)


if __name__ == '__main__':
    agent = MotivationOperator()
    response = agent("Schedule a workout for 10 pm today.")

    print("\n\nFINAL OUTCOMES: ")
    print("reminder:", response)
