import re 
from typing import Optional
from llama import BasicModelRunner
from base_operator import Operator

class PlanningOperator(Operator):
    def __init__(self, model_name: Optional[str], system_prompt: Optional[str], planning_prompt: Optional[str], verbose = True):
        super().__init__()
        
        self.model_prompt_template = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction} [/INST]{cue}"
        self.planner = BasicModelRunner(model_name = model_name or "meta-llama/Llama-2-13b-chat-hf")

        self.create_tools_prompt()
        self.create_planning_prompt_templates(system_prompt, planning_prompt)

        self.enumerated_list_pattern = r'^(?:\d+\.\s+)?(.*?)(?=\d+\.\s+|\Z)'
        self.verbose = verbose

    def create_planning_prompt_templates(self, system_prompt: Optional[str], planning_prompt: Optional[str]):
        self.planner_system_prompt = system_prompt or """You make plans about what actions to take given what information the user provides and what the user requires. Really think if a tool is required. If not, don't use it.
        
Example session:
User: I want to do a workout to feel better.
System:
Plan:
1. Use the tool getRecommendation to suggest a workout for the user.
2. Use the tool scheduleWorkout to set the suggested workout on the user's schedule.

Example session:
User: I weigh 100 pounds and I am 6 feet tall.
System:
Plan:
1. Use the tool setWeight to set the user's weight.
2. Use the tool setHeight to set the user's height."""

        self.planning_suffix = planning_prompt or "Make a multi-step using only the necessary tools. Do not use a tool if not required. Do not explain the logic of planning. In each step, include the chosen tool name or description (no need to specify arguments). Do not ask the user for more inputs. Propose steps on what information is already available and what tools can be used with it."
        planning_tools_template = "Tools available: {tools}\n\nConversation:\n"
        planning_user_query_template = "User: {user_query}\n{planning_suffix}"
        
        self.planning_prompt_template = planning_tools_template + planning_user_query_template
        self.planning_prompt_template_chat_history = planning_tools_template + "{chat_history}\n" + planning_user_query_template
        self.planning_cue = " 1."
        
        self.step_prompt_template = "Chat history: {chat_history}\n\nLatest user message: {query}\n\nAction to take: {step}"

    def create_tools_prompt(self):
        tools_string = ""
        for tool_name, tool_obj in self.operations.items():
            tool_arguments_string = ""
            
            for i, arg in enumerate(tool_obj['arguments']):
                tool_arguments_string += f"{i+1}) {arg['name']} ({arg['type']}): {arg['description']} "
            
            tools_string += f"\n- {tool_name}: {tool_obj['description']}\n{tool_name} has arguments: {tool_arguments_string}"
        return tools_string
    
    def postprocess_enumerated_list(self, text):
        list_items = []
        items = text.split("\n")
        for item in items:
            item = item.strip()
            if re.match(r"^\d+\.", item):
                list_items.append(item)
        return list_items

    def plan(self, user_query, chat_history=None):
        tools = self.create_tools_prompt()
        
        if chat_history is not None:
            instruction_prompt_template = self.planning_prompt_template_chat_history
        else:
            instruction_prompt_template = self.planning_prompt_template
        
        instruction = instruction_prompt_template.format(
            chat_history=chat_history,
            user_query=user_query,
            tools=tools,
            planning_suffix=self.planning_suffix
        )

        prompt = self.model_prompt_template.format(
            system_prompt=self.planner_system_prompt,
            instruction=instruction,
            cue=self.planning_cue
        )

        if self.verbose:
            print(f"[PLAN prompt] {prompt}")
            print(f"[PLAN prompt length] {len(prompt)}")
       
        out = self.planner(str(prompt))
        
        if self.verbose:
            print(f"[PLAN out] {out}")
        
        list_out = self.postprocess_enumerated_list(self.planning_cue + out)
        
        if self.verbose:
            print(f"[PLAN list]: {list_out}")
        
        return list_out
    
    def execute_plan(self, plan, query, chat_history=None):
        prev_obs = []
        for i, step in enumerate(plan):
            if self.verbose:
                print(f"Action #{i+1}: {step}")
            
            prompt = self.step_prompt_template.format(chat_history=chat_history, query=query, step=step)
            
            if prev_obs:
                prev_obs_str = self.list_obs_to_str(prev_obs)
                prompt += f"\n\nPrevious observations: {prev_obs_str}"
            
            obs = self.run(step, prompt)
            prev_obs.append(obs)
            
            if self.verbose:
                print(f"Observation #{i+1}: {obs}")
        
        return prev_obs

    def list_obs_to_str(self, obs_list):
        obs_str = ""
        for i, obs in enumerate(obs_list):
            obs_str += f"{i+1}) {obs}"
            if i != len(obs_list) - 1:
                obs_str += '; '
        return obs_str

    def __call__(self, query, chat_history=None):
        print("Generating plan...")
        plan = self.plan(query, chat_history)
        
        plan_string = ""
        for i, step in enumerate(plan):
            plan_string += f"{step}\n"
        print(f"Plan:\n{plan_string}")
        print(plan)
        
        print("Executing plan...")
        prev_obs = self.execute_plan(plan, query, chat_history)

        all_obs_str = self.list_obs_to_str(prev_obs)
        return f"\nCompleted plan:\n{all_obs_str}"

