import random
from typing import List, Tuple, Dict
import os

# Define types
State = List[List[str]]
Action = Tuple[str, str, str]  # (block, from, to)

def state_to_prompt(state: State, goal: State) -> str:
    """
    Convert a Blocksworld state to a prompt description for the LLM.
    """
    prompt = "Blocksworld Task:\n\nInitial State:\n"
    for stack, blocks in state.items():
        prompt += f"{stack}: {' '.join(blocks)}\n"

    prompt += "\nGoal State:\n"
    for stack, blocks in goal.items():
        prompt += f"{stack}: {' '.join(blocks)}\n"

    prompt += "\nPlease provide a series of moves to reach the goal state. " \
              "You can only move one block at a time. And that box should be the top box of the stack. " \
              "Note that from the left to the right in each stack is the order from the bottom to the top of boxes. " \
              "For example, in stack A B C D, A is the bottom box and D is the top box so that you can only move D in this case. "
    prompt += "***Be careful that you can only pick up the top box in each stack. Check this rule before your move!***. "
    prompt += "\nEach move should be in the format: 'Move [block] from [source] to [destination]'. "
    prompt += "You cannot create new stacks but only move among the existing stacks. "
    prompt += "Separate each move with a newline. Surround the answer with <<<content>>>. "
    prompt += "Answer with the required format like the example: <<<Move B from 2 to table\nMove A from 1 to 2\nMove C from 3 to 1\nMove D from 3 to 2\nMove B from 1 to 2>>>\n"
    prompt += "Your answer: \n"

    return prompt

def create_task(num_blocks: int, num_stacks: int, goal_stacks: int) -> Tuple[State, State]:
    """
    Create a Blocksworld task with a given number of blocks and stacks.
    Returns initial state and goal state.
    """
    blocks = [chr(65 + i) for i in range(num_blocks)]

    # Create initial state
    initial_state = {f"stack{i + 1}": [] for i in range(num_stacks)}
    remaining_blocks = blocks.copy()
    for _ in range(num_blocks):
        block = random.choice(remaining_blocks)
        remaining_blocks.remove(block)
        stack = f"stack{random.randint(1, num_stacks)}"
        initial_state[stack].append(block)

    # Create goal state
    goal_state = {f"stack{i + 1}": [] for i in range(goal_stacks)}
    for block in blocks:
        stack = f"stack{random.randint(1, goal_stacks)}"
        goal_state[stack].append(block)

    return initial_state, goal_state

def write_state_to_file(initial_state: State, goal_state: State, filename: str):
    """
    Write the initial and goal states to a text file.
    """
    with open(filename, 'w') as f:
        f.write("Initial State:\n")
        for stack, blocks in initial_state.items():
            f.write(f"{stack}: {' '.join(blocks)}\n")
        f.write("\nGoal State:\n")
        for stack, blocks in goal_state.items():
            f.write(f"{stack}: {' '.join(blocks)}\n")

def read_state_from_file(filename: str) -> Tuple[State, State]:
    """
    Read the initial and goal states from a text file.
    """
    initial_state = {}
    goal_state = {}
    current_state = initial_state

    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line == "Initial State:":
            current_state = initial_state
        elif line == "Goal State:":
            current_state = goal_state
        elif line:
            parts = line.split(': ')
            stack = parts[0] if len(parts) > 1 else parts[0][:-1]
            blocks = parts[1].split() if len(parts) > 1 else []
            current_state[stack] = blocks

    return initial_state, goal_state

base_dir = '../dataset_gather/Blocksworld_dataset/'
for num_blocks, initial_stacks, goal_stacks in [
    (2, 3, 2), (2, 3, 3), (2, 4, 2), (2, 4, 3), (2, 4, 4), (2, 5, 2), (2, 5, 3), (2, 5, 4),
    (3, 3, 2), (3, 3, 3), (3, 4, 2), (3, 4, 3), (3, 4, 4), (3, 5, 2), (3, 5, 3), (3, 5, 4),
    (4, 3, 2), (4, 3, 3), (4, 4, 2), (4, 4, 3), (4, 4, 4), (4, 5, 2), (4, 5, 3), (4, 5, 4)
]:
            for index in range(5):
                saving_dir = base_dir + f"{num_blocks}_{initial_stacks}_{goal_stacks}_{index}/"
                if not os.path.exists(saving_dir):
                    os.makedirs(saving_dir)

                initial_state, goal_state = create_task(num_blocks, initial_stacks, goal_stacks)

                # Write states to file
                write_state_to_file(initial_state, goal_state, saving_dir + f"blocksworld_task.txt")

                # Read states from file
                read_initial_state, read_goal_state = read_state_from_file(saving_dir + f"blocksworld_task.txt")

                # Verify that the read states match the original states
                print(f"Initial state matches for {num_blocks} blocks, {initial_stacks} initial stacks, {goal_stacks} goal stacks:", initial_state == read_initial_state)
                print(f"Goal state matches for {num_blocks} blocks, {initial_stacks} initial stacks, {goal_stacks} goal stacks:", goal_state == read_goal_state)

                # Generate prompt from the read states
                prompt = state_to_prompt(read_initial_state, read_goal_state)
                print("\nGenerated prompt from file:")
                print(prompt)
