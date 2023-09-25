from llm_operator import Operator
import os
from llama import Lamini

os.environ["LLAMA_ENVIRONMENT"] = "STAGING"

person_age = None


os.environ["LLAMA_ENVIRONMENT"] = "STAGING"
prompt_template = """\
Respond to the message using function calls. You have access to the following tools:

{input:operations}

Use the following format:

Message: the input message you must use
Tool: the tool you will use. Only write the name of the tool

Begin!

Message: {input:query}
Tool: """
operation_selector = Lamini(
    "operator", "meta-llama/Llama-2-13b-chat-hf", prompt_template
)

argument_generator = Lamini(
    "operator",
    "text-davinci-003",  # "meta-llama/Llama-2-13b-chat-hf"
)
prompt_template = """\
You are a helpful assistant. You've just been asked to help with a task with the tools:
{input:operations}
The user's message: {input:query}
You decided to use the tool {input:operation} with the arguments {input:args}
Once you used the tool you got the output {input:output}
Respond to the user's message 

{input:query}

with a final message explaining the actions you took. Do not talk about using tools. Be helpful and inform the user of what has happened.
If the tool's output includes an error, tell the user that there was an error. 
Otherwise, acknowledge the user's message and tell them what you did: 
"""
vocal_llm = Lamini("vocal", "meta-llama/Llama-2-13b-chat-hf", prompt_template)
onboardingOperator = Operator(operation_selector, argument_generator, vocal_llm)
onboardingOperator.add_operation(setAge)
onboardingOperator.add_operation(setHeight)


def callOnboardingAgent(message: str):
    """Use onboarding agent when the user tells you information about themselves"""
    print("calling onboarding agent with message: ", message)
    output = onboardingOperator.run(message, pass_directly=False)
    return output


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


motivationAgent = Operator(operation_selector, argument_generator, vocal_llm)
motivationAgent.add_operation(setReminder)
motivationAgent.add_operation(sendCongrats)
motivationAgent.add_operation(sendMotivationalMessage)


def callMotivationAgent(message: str):
    """Use motivation agent when you want to motivate the user"""
    print("calling motivation agent with message: ", message)
    output = motivationAgent.run(message, pass_directly=False)
    return output


routerOperator = Operator(operation_selector, argument_generator, vocal_llm)
routerOperator.add_operation(callOnboardingAgent)
routerOperator.add_operation(callMotivationAgent)

# print("=====================================================================")
# print("User: I am 19 years old")
# response = routerOperator.run("I am 19 years old")
# print(response)

# print("=====================================================================")
# print("User: This building is 100 years old, and i am just 43.")
# response = routerOperator.run("This building is 100 years old, and i am just 43.")
# print(response)

# print("=====================================================================")
# print("I feel like i can't run fast enough. I just get too tired after a mile :(")
# response = routerOperator.run(
#     "I feel like i can't run fast enough. I just get too tired after a mile :("
# )
# print(response)


print("=====================================================================")
print("Remind me to workout tomorrow at 5pm")
response = routerOperator.run("Remind me to workout tomorrow at 5pm")
print(response)
