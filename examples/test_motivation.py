import os
from llm_operator import Operator
from datetime import date

os.environ["LLAMA_ENVIRONMENT"] = "PRODUCTION"


class MotivationOperator(Operator):
    def setReminder(self, workout_name: str, workout_time: str):
        """
        set a reminder message to the user to do workout.

        Parameters:
        workout_name: name of the workout. if no name given, keep it static at 'no-name'
        workout_time: date and time to schedule the workout.
        """
        print("It is indicated to be a reminder message.")
        return f"Reminder has been set. Workout: {workout_name}, Time: {workout_time}"

    def sendCongratsMessage(self, message: str):
        """
        send a congratulatory message to the user on completing the workout.

        Parameters:
        message: the congratulatory message.
        """
        print("It is indicated to be a congratulatory message.")
        return "Sending user message=" + message

    def sendFollowupMessage(self, message: str):
        """
        send a follow-up message to the user checking on him for missing the workout.

        Parameters:
        message: a message meant to follow up with the user on missing a workout
        """
        print("It is indicated to be a follow up message.")
        return "Sending user message=" + message

    def add_operations(self):
        self.add_operation(self.setReminder)
        self.add_operation(self.sendCongratsMessage)
        self.add_operation(self.sendFollowupMessage)

    def __call__(self, mssg):
        return self.run(mssg)


if __name__ == '__main__':
    # train and  inference
    # #optional training file path
    # training_file = None
    # router_save_path = "examples/models/clf/MotivationOperator/"
    # operator = MotivationOperator()
    # operator.add_operations()
    # operator.train(training_file, router_save_path)
    # query = "Yay you did it. This is awesome!"
    # response = operator(query)

    # inference
    router_save_path = "examples/models/clf/MotivationOperator/router.pkl"
    operator = MotivationOperator().load(router_save_path)
    operator.add_operations()

    query2 = "Yay you did it. This is awesome!"
    print(f"\n\nQuery: {query2}")
    response2 = operator(query2)
    print(response2)
    query3 = "Hey Aaron, hope you are well! I noticed you missed our workout together at Hike in Mt. Abby, Alaska on Monday. It is important to stay consistent with your fitness routine, so I hope you can make it to our next workout together."
    print(f"\n\nQuery: {query3}")
    response3 = operator(query3)
    print(response3)
