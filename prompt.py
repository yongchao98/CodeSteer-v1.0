#### Agent Prompt ####
code_text_choice_prompt = f'''
    You are guiding another TaskLLM to solve a task. You will be presented with a task that can potentially be solved using either pure textual reasoning or coding.
    Your goal is to determine which method will be most effective for solving the task. Follow these steps:

    **Respond** with the chosen approach but not the solution. You can choose between the following options:
       - If you choose coding, respond <<<Code>>> in the end of your response.
       - If you choose textual reasoning, respond <<<Text>>> in the end of your response.

    Now, here is the task:\n
    '''

decision_prompt_text2code = f'''
    Your goal is to give effective prompt guidance for the TaskLLM to help TaskLLM accomplish this task.
    You will be presented with the initial question, the previous prompt guidances and answers from the TaskLLM, the code execution results (if code is generated), the complexity analysis of the code, and the correctness checking results of the answer.

    Return the guidance prompt to help the TaskLLM improve the answer or code.
    **If previous answers use text reasoning, you must generate a prompt to switch to code generation.**
    **Many tasks require highly complex code with symbolic computing, but the TaskLLM generated codes are usually trivial and simple, without any symbolic computing and efficient searching. In this case, try to return the prompt to stimulate TaskLLM to generate better code.
    **Particularly use the returned code complexity analysis as the information. If the code does not use symbolic computing or efficient searching, try to stimulate TaskLLM to generate better code like 'solve this question with more complex and delicate code but not just simply displaying the answer'.
    - If you think the current answer iteration is helpless, then you can directly return <<<Terminate>>>.
    - If you think the TaskLLM's answer or code is correct, you can directly return <<<Return Answer>>>.

    Now, here is the task:\n

'''

decision_prompt_complex_code = f'''
    Your goal is to give effective prompt guidance for the TaskLLM to help TaskLLM accomplish this task.
    You will be presented with the initial question, the previous prompt guidances and answers from the TaskLLM, the code execution results (if code is generated), the complexity analysis of the code, and the correctness checking results of the answer.
    
    Return the guidance prompt to help the TaskLLM improve the answer or code.
    **Many tasks require highly complex code with symbolic computing, but the TaskLLM generated codes are usually trivial and simple, without any symbolic computing and efficient searching. In this case, try to return the prompt to stimulate TaskLLM to generate better code.
    **Particularly use the returned code complexity analysis as the information. If the code does not use symbolic computing or efficient searching, try to stimulate TaskLLM to generate better code like 'solve this question with more complex and delicate code but not just simply displaying the answer'.
    - If you think the current answer iteration is helpless, then you can directly return <<<Terminate>>>.
    - If you think the TaskLLM's answer or code is correct, you can directly return <<<Return Answer>>>.

    Now, here is the task:\n

'''

decision_prompt = f'''
    You are guiding another TaskLLM to solve a task. You will be presented with a task that can potentially be solved using either pure textual reasoning or coding.
    Your goal is to determine effective actions for the TaskLLM to help TaskLLM accomplish this task.
    Follow these steps:
    1) You will be **presented** with the initial question, the previous prompt guidances and answers from the TaskLLM, the code execution results (if code is generated), 
    the complexity analysis of the code, and the correctness checking results of the answer.
    2) **Analyze** the question and the answers from the TaskLLM. Return the guidance prompt to help the TaskLLM improve the answer or code.
    Here are the types of prompts you can generate:
    - If the TaskLLM needs to switch from generating code answer to text answer, then generate a prompt to order the TaskLLM to answer by text.
    - If the TaskLLM needs to switch from generating text answer to code answer, then generate a prompt to order the TaskLLM to answer by code.
    - If the TaskLLM needs to provide a better reasoning answer, generate a prompt to help the TaskLLM improve the answer or generate a answer with different solution thoughts.
    **Especially, many tasks require highly complex code with symbolic computing, but the TaskLLM generated codes are usually trivial and simple, without any symbolic computing and efficient searching. In this case, try to stimulate TaskLLM to generate better code.**
    - **If previous answers all use the same method (text or code), generate a prompt to switch to the other method.**
    - If you think the current answer iteration is helpless, then you can directly return <<<Terminate>>>.
    - If you think the TaskLLM's answer or code is correct, you can directly return <<<Return Answer>>>.

    Now, here is the task:\n
    
'''

with_COT_code_output_prompt = r'''You are a helpful AI assistant. Solve tasks using your coding skills.
    In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    Don't include multiple code blocks in one response, only include one in the response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant.
    Think the task step by step if you need to. If a plan is not provided, explain your plan first. You can first output your thinking steps with texts and then the final python code.
    Remember in the final code you still need to output each number in the final equation!
    Start the python block with ```python

    '''

text_output_prompt = r'''
Analyze the question step by step and try to list all the careful points. Then try to acquire the final answer with step by step analysis.
In the end of your response, directly output the answer to the question. Do not output the code for execution.
    '''

Code_checker_prompt = r'''
Given the following question and the answer from other LLMs, write a python code block to check the correctness of the answer.
Try to generate the code to check the correctness of the answer. Try your best to check whether the answer satisfy all the constraints of the given question.
If the answer is correct, return the text "Correct". If the answer is incorrect, return the reason why the answer is wrong, like what condition or constraint is not satisfied.

'''

multi_turn_planning_prompt_with_code = f'''
    This is the execution results from your code in the last response, please analyze the question and execution results and try to list all the careful points.\n
    Then output the better code based on the errors appearing in previous answers and your analysis.
    \nPut # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant.
    Otherwise, If you feel the plan is completed, please output your final answer and reply "TERMINATE" in the end of your response.

    '''

multi_turn_planning_prompt_without_code = f'''
    This is the answer of you in the last response, please analyze the question and answer, and try to list all the careful points.\n
    Then output the better answer based on the errors appearing in previous answers and your analysis.
    Otherwise, If you feel the plan is completed, please output your final answer and reply "TERMINATE" in the end of your response.

    '''

combined_agent_prompt = f'''You are a helpful AI assistant.
    Solve tasks using your coding and language skills.
    In the following cases, there are two different agents respond to the same problem. In some cases, they output the direct answer, while sometimes they output the code to calculate the answer.
    I will display you the initial question and the answers from two agents. The code execution results will also be given if the code exists.
    Your task is to analyze this question based on the analysis and answers from above two agents and then output your final answer.\n
    If you want to generate code to acquire the answer, suggest python code (in a python coding block) for the user to execute. Don't include multiple code blocks in one response, only include one in the response. 
    Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant.
    I hope you can perform better than other two agents. Hence, try to choose the best answer and propose a new one if you think their methods and answers are wrong.\n'''

AutoGen_prompt = '''You are a helpful AI assistant.
    Solve tasks using your coding and language skills.
    In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
        1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
        2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
    Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
    When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
    If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
    If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
    When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
    Reply "TERMINATE" in the end when everything is done.\n'''