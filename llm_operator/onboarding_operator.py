import os
import argparse
from typing import Optional
import random
from textwrap import dedent

from base_operator import Operator
from base_planning_operator import PlanningOperator
from base_inquiry_operator import InquiryOperator
from llama import BasicModelRunner
from llama import Lamini


os.environ["LLAMA_ENVIRONMENT"] = "PRODUCTION"


class OnboardingOperator(InquiryOperator):
    def __init__(self, model_name: Optional[str] = None, system_prompt: Optional[str] = None,
                 planning_prompt: Optional[str] = None, verbose=False):
        super().__init__(model_name, system_prompt, planning_prompt, verbose=verbose)
        self.age = None
        self.height = None
        self.weight = None
        self.add_operation(self.setAge)
        self.add_operation(self.setHeight)
        self.add_operation(self.setWeight)
        self.add_operation(self.clarify)

        self.chat = BasicModelRunner(model_name="meta-llama/Llama-2-13b-chat-hf")
        self.prompt_template = """<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction} [/INST]{cue}"""

        self.chat_history = None
    
    def set_chat_history(self, chat_history):
        self.chat_history = chat_history

    def get_remaining_values_prompt(self):
        remaining = []
        if not self.age:
            remaining.append("age")
        if not self.height:
            remaining.append("height")
        if not self.weight:
            remaining.append("weight")
        
        # Create prompt on remaining
        remaining_item = random.choice(remaining)
        remaining_prompt = f"Remaining demographic data to ask the user about: {remaining_item}"
        return remaining_prompt

    def isDone(self):
        return self.age is not None and self.height is not None and self.weight is not None

    def clarify(self, operator_response: str):
        """
        clarify user input by asking more questions or replying to their question and asking to complete onboarding tasks of setting weight, height and age.

        Parameters:
        operator_response: response from the operator to user input.
        """
        return f"{operator_response}"

    def setAge(self, age: int):
        """
        given the age of a person, set their age in the system.

        Parameters:
        age: age of the person in years.
        """
        self.age = age
        extra_instruction = f"\n\nConfirm that you set the user's age={age}. "
        return self.respond(extra_instruction)
    
    def respond(self, extra_instruction):
        instruction = f"Chat history:\n{self.chat_history}"
        instruction += extra_instruction
        if self.isDone():
            instruction += "Tell the user that you've collected all the demographic data and they can now start using the app."
        else:
            remaining_prompt = self.get_remaining_values_prompt()
            instruction += f"{remaining_prompt}\nAsk the user about the item in the remaining demographic data. Keep your response brief and a single turn."

        system_prompt = "You are a fitness app onboarding a user. You just collected a user's age information. Respond in 1-2 sentences. Keep your response brief."
        cue = "System: "
        prompt = self.prompt_template.format(system_prompt=system_prompt, instruction=instruction, cue=cue)
        operator_response = self.chat(prompt)
        print(f"PROMPT: {prompt}")
        print(f"OPERATOR RESPONSE: {operator_response}")
        postprocessed_operator_response = operator_response.split('\n')[0].split(' #')[0]
        return postprocessed_operator_response

    def setHeight(self, height: int):
        """
        given the height of a person, set their height in the system.

        Parameters:
        height: height of the person in cms. If the height is in feet and inches, convert it to cm. Eg: 5'6" = 167.
        """
        self.height = height
        extra_instruction = f"\n\nConfirm that you set the user's height={height}. "
        return self.respond(extra_instruction)

    def setWeight(self, weight: int):
        """
        given the weight of a person, set their weight in the system.

        Parameters:
        weight: weight of the person in pounds. If the weight is in any other unit, convert it to lbs.
        """
        self.weight = weight
        extra_instruction = f"\n\nConfirm that you set the user's weight={weight}. "
        return self.respond(extra_instruction)

def train(operator_save_path, training_data=None):
    """Trains the Operator."""
    operator = OnboardingOperator()
    operator.train(operator_save_path, training_data)
    print('Done training!')

# Example session:
# System: Enter your weight, height and age.
# User: I am James. I am 40 years old and weigh 120lbs.
# System: 1. Calling setAge(40).
# 2. Calling setWeight(120, "lbs").
def main():
    llm = BasicModelRunner(model_name="meta-llama/Llama-2-13b-chat-hf")
    prompt_template = """<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction} [/INST]{cue}"""
    
    system_prompt = "You are a fitness app onboarding a user and collecting demographic information. Respond in 1-2 sentences. Keep your response brief."
    cue = "System: "
    instruction = """Chat History:\nSystem: What's your age?\nUser: 43\nSystem: Great, I have noted your age as 43. Now, can you tell me your height?\n
Latest user message: I'm 120 lbs, and 5' 10"""
    prompt = prompt_template.format(system_prompt=system_prompt, instruction=instruction, cue=cue)
    response = llm(prompt)
    print("PROMPT ", prompt)
    print("RESPONSE ", response)

    # Extract with jsonformer
    prompt_template = """\
    <s>[INST] <<SYS>> Extract demographic information: age, weight, height.

    Output format:
    'age': age of the user
    'weight': weight of the user
    'height': height of the user 
    For example: { "age": 43, "weight": 120, "height": 5'10" }
    <</SYS>>

    Given:
    'Chat history': {input:chat_history}
    'Demographic data': {input:demographic_data}
    'Latest user message': {input:latest_user_message}

    Extract and update the demographic data only. Do not explain the logic.
    [/INST] """

    # { "age": 43, "weight": 120, "height": 5'10" }
    demographic_data = '{ "age": None, "weight": None, "height": None }'
    prompt_template = dedent(prompt_template)
    extractor_llm = Lamini("structured", "meta-llama/Llama-2-13b-chat-hf", prompt_template)
    output_type = { "age": "str", "weight": "str", "height": "str" }
    latest_user_message = "I'm 95 kg, and 6ft 1"
    chat_history = "System: What's your age?\nUser: 40\nSystem: Great, I have noted your age as 43. Now, can you tell me your height?\n"
    input = {
        "chat_history": chat_history,
        "demographic_data": demographic_data,
        "latest_user_message": str(latest_user_message)
    }
    model_response = extractor_llm(
            input,
            output_type,
            stop_tokens=["</s>"]
        )
    print("MODEL RESPONSE ", model_response)

    import pdb;pdb.set_trace()
    args = argparse.ArgumentParser()
    args.add_argument("--verbose", action="store_true", help="Print extra info", default=False)
    args = args.parse_args()

    operator_save_path = "models/OnboardingOperator/"
    # model_name = "gpt-4"
    # train(operator_save_path)
    system_prompt = """The system asks a question to get user details. Use the use input to decide which tool to use for utilising the user information. Ask clarifying questions if something is unclear.
Example session:
System: Enter your weight, height and age.
User: my age is 50.
System: I'll use the tool setAge to set the user's age to 50."""

    planning_prompt = "Use the latest user message along with chat history to decide which tool(s) to use to act on user information."

    operator = OnboardingOperator(model_name=None,
                                  system_prompt=system_prompt,
                                  planning_prompt=planning_prompt,
                                  verbose=args.verbose
                                  ).load(operator_save_path)

    user_onboard_trigger = "I'd like to get started"
    onboarding_prompt = """Sure, I'd be happy to help you get started. I'll need to ask you a few questions to set up your profile.
To start, what's your age?"""
    # print("Enter your age (in years): ")
    chat_history = f"User: {user_onboard_trigger}\n"
    operator_history = f"System: {onboarding_prompt}\n"
    
    chat_history += operator_history
    user_response = input(operator_history)
    
    while not operator.isDone():
        chat_history += f"User: {user_response}\n"
        operator.set_chat_history(chat_history)
        
        operator_response = operator(user_response, chat_history)    
        operator_history = f"System: {operator_response}\n"

        chat_history += operator_history
        print(operator_history)

        if operator.isDone():
            break 
        user_response = input("User: ")


if __name__ == '__main__':
    main()
