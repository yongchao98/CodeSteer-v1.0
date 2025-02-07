import os
import json
import io
import sys
import time
import openai
from openai import OpenAI
import tiktoken
import re
import numpy as np
from prompt import Code_checker_prompt
import subprocess
import anthropic
from mistralai import Mistral
from typing_extensions import override
from openai import AssistantEventHandler
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

# Access the keys safely
openai_api_key = os.getenv('OPENAI_API_KEY')
claude_api_key = os.getenv('CLAUDE_API_KEY')
mixtral_api_key = os.getenv('MIXTRAL_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')

def save_dataset_item(dataset_item, json_file_path="dataset.json"):
    """
    Appends a single dataset_item to an existing JSON list or creates a new JSON list if the file doesn't exist.
    """
    # If file exists, read the existing data; otherwise, create an empty list.
    if os.path.exists(json_file_path):
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                # If the existing JSON is not a list, overwrite with a list
                data = []
    else:
        data = []

    # Append the new item
    data.append(dataset_item)
    print(f'Length of DPO dataset: {len(data)}')

    # Write everything back to the file
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def log_run_info(log_file, run_info):
    with open(log_file, 'a') as f:
        f.write(run_info + "\n")

def transform_item_to_messages(item):
    """
    Transform a dataset item into a list of messages with user/assistant roles.

    Args:
        item: Dictionary containing 'instruction', 'output', and 'history'

    Returns:
        list: List of message dictionaries with roles and content
    """
    messages = []

    # First add all history items
    for hist_input, hist_output in item['history']:
        messages.extend([
            {"role": "user", "content": hist_input},
            {"role": "assistant", "content": hist_output}
        ])

    # Then add the final instruction and output
    messages.extend([
        {"role": "user", "content": item['instruction']}
    ])

    #messages.extend([
    #    {"role": "user", "content": item['instruction']},
    #    {"role": "assistant", "content": item['output']}
    #])
    return messages

def message_input_process_codellama_qven(chat):
    output = "<s>"
    for m in chat:
        output += f"Source: {m['role']}\n\n {m['content'].strip()}"
        output += " <step> "
    output += "Source: assistant\nDestination: user\n\n "
    messages = [{"role": "user", "content": output}]
    return messages

def paraphrase_with_GPT4(statement):
    prompt = 'Paraphrase the following instruction with the semantic meanings keeping the same. Directly output the paraphrased version without any other contents.\n\nOriginal instruction: '

    new_statement = GPT_response('', prompt + statement, model_name='gpt-4o', code_interpreter=False, user_prompt_list=[prompt + statement], response_total_list=[], logprobs=False)
    return new_statement

def save_file_func(save_code_dir, response_list, user_prompt_list, question,
                   CodeSteer_input_prompt_list, CodeSteer_input_prompt_training_list,
                   CodeSteer_output_prompt_guidance_list):
    """
    Save all conversation data to a JSON file.

    Args:
        save_code_dir (str): Directory path to save the file
        response_list (list): List of responses
        user_prompt_list (list): List of user prompts
        question (str): The original question
        CodeSteer_input_prompt_list (list): List of CodeSteer input prompts
        CodeSteer_input_prompt_training_list (list): List of CodeSteer training prompts
        CodeSteer_output_prompt_guidance_list (list): List of CodeSteer guidance prompts
    """
    data = {
        'question': question,
        'response_list': response_list,
        'user_prompt_list': user_prompt_list,
        'CodeSteer_input_prompt_list': CodeSteer_input_prompt_list,
        'CodeSteer_input_prompt_training_list': CodeSteer_input_prompt_training_list,
        'CodeSteer_output_prompt_guidance_list': CodeSteer_output_prompt_guidance_list
    }

    output_file = os.path.join(save_code_dir, 'conversation_data.json')

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving data: {str(e)}")


def load_conversation_data(file_path):
    """
    Load conversation data from a JSON file and return the original list formats.

    Args:
        file_path (str): Path to the JSON file

    Returns:
        tuple: (response_list, user_prompt_list, question, CodeSteer_input_prompt_list,
               CodeSteer_input_prompt_training_list, CodeSteer_output_prompt_guidance_list)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return (
            data['response_list'],
            data['user_prompt_list'],
            data['question'],
            data['CodeSteer_input_prompt_list'],
            data['CodeSteer_input_prompt_training_list'],
            data['CodeSteer_output_prompt_guidance_list']
        )
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def extract_and_check(response):
    matches = re.findall(r'<<<(.*?)>>>', response, re.DOTALL)
    extracted_text = matches[-1] if matches else ''
    itertools_present = 'code_interpreter' in response or '\n```python' in response
    return extracted_text, itertools_present

def extract_code(text):
    # Regular expression to match code blocks enclosed in triple backticks
    code_block_pattern = re.compile(r'```python\n(.*?)\n```', re.DOTALL)

    # Find all matches in the text
    code_blocks = code_block_pattern.findall(text)

    # If no code blocks are found, try to find indented code blocks
    if not code_blocks:
        return []

    return code_blocks


def LLM_answer_code_checker(question, response, check_code_saving_path):
    Input_prompt = f'{Code_checker_prompt}\nThe question is:\n{question}\nThe response is:\n{response}\nYour checking code:\n'
    user_prompt_list = [Input_prompt]; response_list = []
    max_retries = 3

    for attempt in range(max_retries):
        code_block_list = []
        # If no code blocks found, retry up to 2 more times
        for iteration in range(5):
            response_check = GPT_response('', Input_prompt, model_name='gpt-4o', code_interpreter=False,
                                          user_prompt_list=user_prompt_list, response_total_list=response_list, logprobs=False)
            code_block_list = extract_code(response_check)
            if len(code_block_list) > 0:
                break

        if len(code_block_list) > 0:
            with open(check_code_saving_path, "w") as f:
                f.write(code_block_list[0])

            try:
                result = subprocess.run(
                    ["python3", check_code_saving_path],
                    capture_output=True, text=True, timeout=10
                )
                output = result.stdout
                errors = result.stderr

                # If no errors, return the output
                if not errors:
                    return f'output: {output}'

            except subprocess.TimeoutExpired as e:
                output = e.stdout if e.stdout else ""
                errors = e.stderr if e.stderr else ""
                timeout_error = f"\nTimeoutExpired: Command '{e.cmd}' timed out after {e.timeout} seconds"
                errors += timeout_error
        else:
            # If no code blocks are found
            errors = f"\nPrevious attempt failed to generate code. Please try again."
            return f'errors: {errors}'

    # If we've exhausted all retries, return the last output and errors
    if isinstance(output, str):
        if count_total_tokens([output], []) > 2000:
            return f'Check function does not work this round.'
        else:
            return f'output: {output}\nerrors: {errors}'
    else:
        return f'errors: {errors}'

def message_input_process_codellama_qven(chat):
    output = "<s>"
    for m in chat:
        output += f"Source: {m['role']}\n\n {m['content'].strip()}"
        output += " <step> "
    output += "Source: assistant\nDestination: user\n\n "
    messages = [{"role": "user", "content": output}]
    return messages

def count_total_tokens(user_prompt_list, response_total_list, model_name="gpt-3.5-turbo"):
    # Initialize the tokenizer
    encoding = tiktoken.encoding_for_model(model_name)

    # Count tokens in each list
    user_prompt_tokens = sum(len(encoding.encode(prompt)) for prompt in user_prompt_list)
    response_total_tokens = sum(len(encoding.encode(response)) for response in response_total_list)

    # Calculate the total token count
    total_tokens = user_prompt_tokens + response_total_tokens

    return total_tokens

def message_construct_func(user_prompt_list, response_total_list, system_message, model_name):
    if model_name in ['o1', "o1-preview", 'o1-mini', 'gpt-4o', 'gpt-4o-mini', 'gpt-35-turbo-16k-0613', 'gpt-4-turbo', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo',
                      'open-mixtral-8x7b', "mistral-large-latest", 'DeepSeek-R1']:
        messages = [{"role": "system", "content": system_message}]
    else:
        messages = []

    if len(user_prompt_list) - len(response_total_list) == 1:
        for i in range(len(user_prompt_list)):
            messages.append({"role": "user", "content": user_prompt_list[i]})
            if i < len(user_prompt_list) - 1:
                messages.append({"role": "assistant", "content": response_total_list[i]})
        return messages
    else:
        raise ValueError(f"The number of user prompts and responses must differ by exactly 1\nlen(user_prompt_list):{len(user_prompt_list)}, len(response_total_list):{len(response_total_list)}")

def convert_messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message["role"] == "user":
            prompt += f"User: {message['content']}\n\n"
        elif message["role"] == "assistant":
            prompt += f"Assistant: {message['content']}\n\n"
        elif message["role"] == "system":
            prompt += f"System: {message['content']}\n\n"
    return prompt.strip()

def message_construct_llama_func(user_prompt_list, response_total_list):
    messages = []
    for i in range(len(user_prompt_list)):
        messages.append({"role": "user", "content": user_prompt_list[i]})
        if i < len(user_prompt_list) - 1:
            messages.append({"role": "assistant", "content": response_total_list[i]})
    return messages


def GPT_response(system_message, question, model_name, code_interpreter, user_prompt_list, response_total_list, logprobs, max_tokens_num = 2000, temperature=0.0):

    for iteration_num in range(25):
        try:
            response = GPT_response_once(system_message, question, model_name, code_interpreter, user_prompt_list, response_total_list, logprobs=logprobs, max_tokens_num=max_tokens_num, temperature=temperature)
            return response
        except Exception as e:
            print(f"Error on iteration {iteration_num + 1}: {e}")
            print("Waiting for 20 seconds before retrying...")
            time.sleep(1)
    raise RuntimeError("Failed to get a response after 10 attempts")

def GPT_response_once(system_message, question, model_name, code_interpreter, user_prompt_list, response_total_list, logprobs, max_tokens_num, temperature):
    openai_api_key_name = openai_api_key
    claude_api_key_name = claude_api_key
    mixtral_api_key_name = mixtral_api_key

    if model_name not in ["o1", "o1-preview", 'o1-mini', 'gpt-4o', 'gpt-4o-mini', 'gpt-35-turbo-16k-0613', 'gpt-4-turbo', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo',
                          "claude-3-5-sonnet-20241022", "claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307",
                          'open-mixtral-8x7b', "mistral-large-latest",
                          'CodeLlama-34b', 'CodeLlama-70b', 'Qwen2.5-32B',
                          'DeepSeek-R1']:
        print('\nModel name is wrong!')
        raise ValueError("Invalid model name!")
    if model_name == 'gpt-35-turbo-16k-0613':
        model_name = 'gpt-3.5-turbo'

    input_messages = message_construct_func(user_prompt_list, response_total_list, system_message, model_name)

    if code_interpreter == False:
        if model_name in ['gpt-4o', 'gpt-4o-mini', 'gpt-35-turbo-16k-0613', 'gpt-4-turbo', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo']:
           client = OpenAI(api_key=openai_api_key_name)
           response = client.chat.completions.create(
                model=model_name,
                messages=input_messages,
                max_tokens=max_tokens_num,
                temperature=temperature,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                logprobs=logprobs
            )

           if logprobs:
               logprobs = [token.logprob for token in response.choices[0].logprobs.content]
               response_text = response.choices[0].message.content
               response_text_tokens = [token.token for token in response.choices[0].logprobs.content]
               perplexity_score = np.exp(-np.mean(logprobs))
               return response_text, perplexity_score, response_text_tokens, logprobs
           else:
               return response.choices[0].message.content

        elif model_name in ['o1', "o1-preview", 'o1-mini']:
            client = OpenAI(api_key=openai_api_key_name)
            response = client.chat.completions.create(
                model=model_name,
                #max_tokens=max_tokens_num,
                messages=input_messages
            )
            return response.choices[0].message.content
        elif model_name in ["claude-3-5-sonnet-20241022", "claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307"]:
            client = anthropic.Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key=claude_api_key_name,
            )

            message = client.messages.create(
                model=model_name,
                # claude-3-sonnet-20240229, claude-3-opus-20240229, claude-3-haiku-20240307
                max_tokens=max_tokens_num,
                temperature=0.0,
                top_p=1,
                system=system_message,
                messages=input_messages,
            )
            return message.content[0].text
        elif model_name in ['DeepSeek-R1']:
            client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=input_messages,
                stream=False
            )
            return response.choices[0].message.content
        elif model_name in ['open-mixtral-8x7b', "mistral-large-latest"]:
            client = Mistral(api_key = mixtral_api_key_name)
            chat_response = client.chat.complete(
                model=model_name,
                messages=input_messages,
                max_tokens=max_tokens_num,
                temperature=temperature,
                top_p=1,
            )
            return chat_response.choices[0].message.content

    elif code_interpreter == True:
        client = OpenAI(api_key=openai_api_key_name)

        # Create a StringIO object to capture the output
        captured_output = io.StringIO()

        # Save the current stdout so we can restore it later
        original_stdout = sys.stdout

        try:
            # Redirect stdout to the StringIO object
            sys.stdout = captured_output

            assistant = client.beta.assistants.create(
                instructions=system_message,
                model=model_name,
                tools=[{"type": "code_interpreter"}],
                temperature=temperature,
                top_p=1
            )

            thread = client.beta.threads.create()

            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=question)

            # EventHandler class to handle the events in the response stream.
            class EventHandler(AssistantEventHandler):
                @override
                def on_text_created(self, text) -> None:
                    print(f"\nassistant > ", end="", flush=True)

                @override
                def on_text_delta(self, delta, snapshot):
                    print(delta.value, end="", flush=True)

                def on_tool_call_created(self, tool_call):
                    print(f"\nassistant > {tool_call.type}\n", flush=True)

                def on_tool_call_delta(self, delta, snapshot):
                    if delta.type == 'code_interpreter':
                        if delta.code_interpreter.input:
                            print(delta.code_interpreter.input, end="", flush=True)
                        if delta.code_interpreter.outputs:
                            print(f"\n\noutput >", flush=True)
                            for output in delta.code_interpreter.outputs:
                                if output.type == "logs":
                                    print(f"\n{output.logs}", flush=True)

            # Use the `stream` SDK helper with the `EventHandler` class to create the Run and stream the response.
            with client.beta.threads.runs.stream(
                    thread_id=thread.id,
                    assistant_id=assistant.id,
                    instructions="",
                    event_handler=EventHandler(),
            ) as stream:
                stream.until_done()

        finally:
            # Reset stdout to the original value
            sys.stdout = original_stdout

        # Get the captured output as a string
        output = captured_output.getvalue()
        return output
    else:
        print('\nCode interpreter expression is wrong!')
        raise ValueError("Invalid model name!")