import re
from typing import Optional
from llama import BasicModelRunner
from base_operator import Operator


class InquiryOperator(Operator):
    def __init__(self, model_name: Optional[str], system_prompt: Optional[str], planning_prompt: Optional[str],
                 verbose=True):
        super().__init__()

        self.model_prompt_template = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction} [/INST]{cue}"
        self.planner = BasicModelRunner(model_name=model_name or "meta-llama/Llama-2-13b-chat-hf")
        # self.planner = BasicModelRunner(model_name=model_name or "mistralai/Mistral-7B-Instruct-v0.1")
        # self.planner = BasicModelRunner(model_name=model_name or "gpt-4")
        planning_tools_template = "Tools available: {tools}\n\nConversation:\n"
        planning_user_query_template = "{planning_suffix}"
        self.planning_prompt_template_chat_history = planning_tools_template + "{chat_history}\n" + planning_user_query_template
        self.system_prompt = system_prompt
        self.planning_prompt = planning_prompt
        self.planning_cue = " System: [PLAN] 1. "
        self.verbose = verbose
        self.step_prompt_template = "Chat history: {chat_history}\n\nLatest user message: {query}\n\nAction to take: {step}"

    def create_tools_prompt(self):
        tools_string = ""
        for tool_name, tool_obj in self.operations.items():
            tool_arguments_string = ""

            for i, arg in enumerate(tool_obj['arguments']):
                tool_arguments_string += f"{i + 1}) {arg['name']} ({arg['type']}): {arg['description']} "

            tools_string += f"\n- {tool_name}: {tool_obj['description']}\n{tool_name} has arguments: {tool_arguments_string}"
        return tools_string

    def postprocess_enumerated_list(self, text):
        list_items = []
        items = text.split("\n")
        for item in items:
            item = item.strip()
            if item.startswith('User:'):
                break
            if len(item) > 0:
                list_items.append(item)
            # if re.match(r"^\d+\.", item):
            #     list_items.append(item)
        return list_items

    def plan(self, user_query, chat_history=None):
        tools = self.create_tools_prompt()

        instruction = self.planning_prompt_template_chat_history.format(
            chat_history=chat_history,
            user_query=user_query,
            tools=tools,
            planning_suffix=self.planning_prompt
        )

        prompt = self.model_prompt_template.format(
            system_prompt=self.system_prompt,
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
        answers = []
        for i, step in enumerate(plan):
            if self.verbose:
                print(f"Action #{i + 1}: {step}")

            prompt = self.step_prompt_template.format(chat_history=chat_history, query=query, step=step)
            print(f"execute_plan: {prompt}")

            print(f"step: {step}, prompt: {prompt}")
            obs = self.run(step, prompt)
            answers.append(obs)
        return ''.join(answers)

    def __call__(self, query, chat_history=None):
        print("Generating plan...")
        plan = self.plan(query, chat_history)

        plan_string = ""
        for i, step in enumerate(plan):
            plan_string += f"{step}\n"
        print(f"Plan:\n{plan_string}")

        print("Executing plan...")
        prev_obs = self.execute_plan(plan, chat_history)
        return str(prev_obs)

