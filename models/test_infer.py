from src.llamafactory.chat.chat_model import run_response

def infer_func(messages, args_path):
    response_answer = run_response(messages, args_path)
    return response_answer

chat = [
    {"role": "system", "content": "You are a python programmer."},
    {"role": "user", "content": "Write python code to form integers 3 3 8 8 into equation of 24"},
    {"role": "assistant", "content": "3 * 3 * 8 * 8 = 24"},
    {"role": "user", "content": "This equation is wrong. Please try again."},
]

output = "<s>"
for m in chat:
    output += f"Source: {m['role']}\n\n {m['content'].strip()}"
    output += " <step> "
output += "Source: assistant\nDestination: user\n\n "

messages = [{"role": "user", "content": output}]
args_path = 'examples/inference/qwen2_code.yaml'

response = infer_func(messages, args_path)
print(response)