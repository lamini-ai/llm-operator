import os
from llm_operator import Operator
from datetime import date
# os.environ["LLAMA_ENVIRONMENT"] = "PRODUCTION"


class MotivationOperator(Operator):
    def setReminder(self, workout_name: str, workout_time: str):
        """
        set a reminder message to the user to do workout.

        Parameters:
        workout_name: name of the workout. if no name given, keep it static at 'no-name'
        workout_time: time to schedule the workout. Figure out the given date and time and convert in ISO datetime format.
        """
        print("setReminder: ")
        return f"Reminder has been set. Workout: {workout_name}, Time: {workout_time}"

    def sendCongratulationsMessage(self, message: str):
        """
        send a congratulatory message to the user on completing the workout.

        Parameters:
        message: the congratulatory message.
        """
        return "congrats:" + message

    def sendFollowupMessage(self, message: str):
        """
        send a follow-up message to the user checking on him for missing the workout.

        Parameters:
        message: a message meant to follow up with the user on missing a workout
        """
        return "followup:" + message

    def __call__(self, mssg):
        self.add_operation(self.setReminder)
        self.add_operation(self.sendCongratulationsMessage)
        self.add_operation(self.sendFollowupMessage)
        return self.run(mssg)


if __name__ == '__main__':
    agent = MotivationOperator("MotivationOperator", "examples/models/clf/MotivationOperator/clf_classes_prompts.json", "examples/models/clf/MotivationOperator")
    query = "Schedule a workout for 6 pm today."
    # query = "Send this message to the user 'Yay you did it. That's awesome.'"
    # query = "Send this message to the user 'Hey Aaron, hope you're doing well! I noticed you missed our workout together at Crow Pass Hike in Alyeska, Alaska on Monday. It's important to stay consistent with your fitness routine, so I hope you can make it to our next workout together. Let me know if you need any help or motivation!'"
    response = agent(query)
    print(response)

