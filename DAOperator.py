# from llama import Lamini
from typing import Callable, Dict, List, Optional, Any

from llm_operator import Operator


class DecisionOperator(Operator):
    def __init__(self):
        self.prompt = '''
        You're a decision agent. Your task is to choose 1 or more tools from the given tool which signify what action to take from the given user input.
        Tools available to you:
        {input:tools}
        Eg:
        Given tools ['Book an appointment','Cancel an appointment', 'Emergency appointment', 'Share patient update', 'Logistical inquiry', 'Financials and payment', 'Other'],
        For user input: 'Hey, can you squeeze me in to doc's schedule?', tools selected should be ['Book an appointment'],
        For user input: 'I'm feeling a lot better. Can you cancel my followup? ['Share patient update', 'Cancel an appointment'],
        For user input: 'I feel a striking pain in my chest. ['Share patient update', 'Emergency appointment']
        '''

