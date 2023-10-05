import re 

from llama import BasicModelRunner
from base_operator import Operator


class PlanningOperator(Operator):
    def __init__(self, verbose=True):
        super().__init__()
        
        self.model_prompt_template = "<s>[INST] <<SYS>>{system_prompt}<</SYS>>{instruction}[/INST]{cue}"
        self.planner = BasicModelRunner(model_name="meta-llama/Llama-2-7b-chat-hf")
    
        self.create_tools_prompt()
        self.create_planning_prompt_templates()

        self.enumerated_list_pattern = r'\d+\.\s(.*?)(?=\s*\d+\.\s|\Z)'
        self.verbose = verbose

    def create_planning_prompt_templates(self):
        self.planner_system_prompt = "You make plans about what actions to take, given a user query and the current state of the conversation. Provide 3 steps on what tools need to be used, given the tools available."
        self.planning_suffix = "\nMake a 3-step plan in an enumerated list, with the tools available. In each step, include the tool or description of using the tool (no need to specify arguments)."
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
        items = [item.strip() for item in re.findall(self.enumerated_list_pattern, text, re.DOTALL)]
        return items

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
        
        out = out.split("\n\n")[0]
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
            plan_string += f"{i+1}) {step}\n"
        print(f"Plan:\n{plan_string}")
        
        print("Executing plan...")
        prev_obs = self.execute_plan(plan, query, chat_history)

        all_obs_str = self.list_obs_to_str(prev_obs)
        return f"\nCompleted plan:\n{all_obs_str}"

