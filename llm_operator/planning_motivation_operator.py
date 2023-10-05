import os
import argparse
import re 

from llama import BasicModelRunner
from motivation_operator import MotivationOperator


os.environ["LLAMA_ENVIRONMENT"] = "PRODUCTION"
os.environ["LLAMA_ENVIRONMENT"] = "STAGING"


class PlanningMotivationOperator(MotivationOperator):
    def __init__(self):
        super().__init__()
        
        self.llama2_prompt_template = "<s>[INST] <<SYS>>{system_prompt}<</SYS>>{instruction}[/INST]{cue}"
        self.model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.enumerated_list_pattern = r'\d+\.\s(.*?)(?=\s*\d+\.\s|\Z)'
        
        self.planner_system_prompt = "You make plans about what actions to take, given a user query and the current state of the conversation. Provide 3 steps on what tools need to be used, given the tools available."
        self.planner = BasicModelRunner(model_name=self.model_name)
    
        self.tools_string = ""
        for tool_name, tool_obj in self.operations.items():
            tool_arguments_string = ""
            for i, arg in enumerate(tool_obj['arguments']):
                tool_arguments_string += f"{i+1}) {arg['name']} ({arg['type']}): {arg['description']} "
            self.tools_string += f"\n- {tool_name}: {tool_obj['description']}\n{tool_name} has arguments: {tool_arguments_string}"

        self.planning_suffix = "\nMake a 3-step plan in an enumerated list, with the tools available. In each step, include the tool or description of using the tool (no need to specify arguments)."
        self.planning_tools_template = "Tools available: {self.tools_string}\n\nConversation:\n"
        self.planning_user_query_template = "User: {user_query}\n{self.planning_suffix}"
        self.planning_prompt_template = self.planning_tools_template + self.planning_user_query_template
        self.planning_prompt_template_chat_history = self.planning_tools_template + "{chat_history}\n" + self.planning_user_query_template
        self.planning_cue = " 1."

    def postprocess_enumerated_list(self, text):
        items = [item.strip() for item in re.findall(self.enumerated_list_pattern, text, re.DOTALL)]
        return items

    def plan(self, chat_history, user_query):
        if chat_history is not None:
            prompt = self.planning_prompt_template_chat_history.format(**locals())
        else:
            prompt = self.planning_prompt_template.format(**locals())
        prompt = self.llama2_prompt_template.format(system_prompt=self.planner_system_prompt, instruction=prompt, cue=self.planning_cue)
        print(f"[PLAN] Prompt: {prompt}")
        print(f"[PLAN] Prompt length: {len(prompt)}")
        out = self.planner(str(prompt))
        print(out)
        print(f"[PLAN] Out: {out}")
        out = out.split("\n\n")[0]
        list_out = self.postprocess_enumerated_list(self.planning_cue + out)
        print(f"[PLAN] Enumerated list: {list_out}")
        return list_out

    def __get_operation_to_run(self, output):
        '''
        Get the tool callback from the name of the tool.
        '''
        for name, val in self.operations.items():
            if output == name:
                return val
            
    def run(self, action: str, context: str):
        '''
        Override the run method to customize the prompt for call to LLMRouter to select operation vs. call to LLM select arguments.
        '''
        if not self.model_load_path:
            raise Exception("Router not loaded.")
        
        selected_operation = self.select_operations(action)
        print(f"selected operation: {selected_operation}")
        generated_arguments = self.select_arguments(context, selected_operation)
        print(f"inferred arguments: {generated_arguments}")
        action = self.__get_operation_to_run(selected_operation)["action"]
        if generated_arguments:
            tool_output = action(**generated_arguments)
        else:
            tool_output = action()
        return tool_output

    def list_obs_to_str(self, obs_list):
        obs_str = ""
        for i, obs in enumerate(obs_list):
            obs_str += f"{i+1}) {obs}"
            if i != len(obs_list) - 1:
                obs_str += '; '
        return obs_str

    def __call__(self, chat_history, query):
        plan = self.plan(chat_history, query)
        prev_obs = []
        for i, step in enumerate(plan):
            print(f"[CALL #{i+1}] Step: {step}")
            prompt = f"Chat history: {chat_history}\n\nLatest user message: {query}\n\nAction to take: {step}"
            if prev_obs:
                prev_obs_str = self.list_obs_to_str(prev_obs)
                prompt += f"\n\nPrevious observations: {prev_obs_str}"
                print(f"[CALL #{i+1}] Observation IN: {prev_obs_str}")
            print(f"[CALL #{i+1}] Prompt: {prompt}")
            obs = self.run(step, prompt)
            prev_obs.append(obs)
            print(f"[CALL #{i+1}] Observation OUT: {obs}")
        
        all_obs_str = self.list_obs_to_str(prev_obs)
        response = f"Completed all steps, with observations: {all_obs_str}"
        return response


def main():
    operator_save_path = "models/MotivationOperator/"
    operator = PlanningMotivationOperator().load(operator_save_path)
    
    chat_history = """User: Hi, I'm feeling down
System: I'm sorry to hear that. What would you like to do?"""
    query = "I want to do a workout to feel better"
    response = operator(chat_history, query)
    print(response)


if __name__ == '__main__':
    main()