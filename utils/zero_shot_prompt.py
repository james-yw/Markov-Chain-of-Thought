zero_shot_prompt_prefix = """You will be provided with an original math problem and its intermediate thought, action and observation. Please generate a sub question that should utilize the results of the intermediate thought, action and observation to replace the related information of original problem and delete redundant irrelevant information. The sub question should be wrapped by <subquestion> and </subquestion>.
<original problem> 
{origin_problem}
</original problem> 
<intermediate thought> 
{thought}
</intermediate thought> 
<intermediate action>
{action}
</intermediate action> 
<intermediate observation> 
{observation}
</intermediate observation>"""