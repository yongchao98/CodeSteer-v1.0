import os
import random
import json
from typing import List, Tuple

# ----------------------
# 1. Define the Pool of 20 Rules
# ----------------------
RULES_POOL = [
    # Rule 1
    (
        "If the string prefix is 'ab', replace it with 'ca'.",
        lambda s: ("ca" + s[2:], 1) if s.startswith("ab") else (s, 0)
    ),
    # Rule 2
    (
        "If the string suffix is 'ac', replace it with 'cb'.",
        lambda s: (s[:-2] + "cb", 2) if s.endswith("ac") else (s, 0)
    ),
    # Rule 3
    (
        "If the string prefix is 'bc', delete the first two characters and append 'aa' to the end.",
        lambda s: (s[2:] + "aa", 3) if s.startswith("bc") else (s, 0)
    ),
    # Rule 4
    (
        "If the string suffix is 'bb', delete the last two characters.",
        lambda s: (s[:-2], 4) if s.endswith("bb") else (s, 0)
    ),
    # Rule 5
    (
        "If the string prefix is 'cb', replace it with 'aa' and delete the last character.",
        lambda s: ("aa" + s[2:-1], 5) if s.startswith("cb") and len(s) > 1 else (s, 0)
    ),
    # Rule 6
    (
        "If the string prefix is 'ca', replace it with 'bb' and append 'c' to the end.",
        lambda s: ("bb" + s[2:] + "c", 6) if s.startswith("ca") else (s, 0)
    ),
    # Rule 7
    (
        "If the string suffix is 'cc', replace it with 'b' and prepend 'a' to the start.",
        lambda s: ("a" + s[:-2] + "b", 7) if s.endswith("cc") else (s, 0)
    ),
    # Rule 8
    (
        "If the string prefix is 'aa', remove the first character.",
        lambda s: (s[1:], 8) if s.startswith("aa") else (s, 0)
    ),
    # Rule 9
    (
        "If the string contains 'abc', replace the first occurrence with 'cab'.",
        lambda s: (s.replace("abc", "cab", 1), 9) if "abc" in s else (s, 0)
    ),
    # Rule 10
    (
        "If the string contains 'bca', delete the first occurrence entirely.",
        lambda s: (s.replace("bca", "", 1), 10) if "bca" in s else (s, 0)
    ),
    # Rule 11
    (
        "If the string ends with 'ba', replace it with 'ab'.",
        lambda s: (s[:-2] + "ab", 11) if s.endswith("ba") else (s, 0)
    ),
    # Rule 12
    (
        "If the string starts with 'cc', remove the first two characters.",
        lambda s: (s[2:], 12) if s.startswith("cc") else (s, 0)
    ),
    # Rule 13
    (
        "If the string contains 'acb', replace the first occurrence with its reverse ('bca').",
        lambda s: (s.replace("acb", "bca", 1), 13) if "acb" in s else (s, 0)
    ),
    # Rule 14
    (
        "If the string ends with 'ca', remove the last character.",
        lambda s: (s[:-1], 14) if s.endswith("ca") and len(s) > 0 else (s, 0)
    ),
    # Rule 15
    (
        "If the string starts with 'bb', remove the second character.",
        lambda s: (s[0] + s[2:], 15) if s.startswith("bb") and len(s) >= 2 else (s, 0)
    ),
    # Rule 16
    (
        "If the string ends with 'aa', replace it with 'cc'.",
        lambda s: (s[:-2] + "cc", 16) if s.endswith("aa") else (s, 0)
    ),
    # Rule 17
    (
        "If the string contains 'ca' (not at the start), remove the first occurrence found after the first character.",
        lambda s: (s[:idx] + s[idx+2:], 17) if (idx := s.find("ca", 1)) != -1 else (s, 0)
    ),
    # Rule 18
    (
        "If the string contains an even number of 'b's (and at least one 'b'), append 'ab' at the end.",
        lambda s: (s + "ab", 18) if (s.count("b") > 0 and s.count("b") % 2 == 0) else (s, 0)
    ),
    # Rule 19
    (
        "If the string length is greater than 15, remove the middle character.",
        lambda s: (s[:len(s)//2] + s[len(s)//2+1:], 19) if len(s) > 15 else (s, 0)
    ),
    # Rule 20
    (
        "If the string starts with 'ac', replace the first two characters with 'zz'.",
        lambda s: ("zz" + s[2:], 20) if s.startswith("ac") else (s, 0)
    ),
]

# ----------------------
# 2. Dynamic String Transformation Functions
# ----------------------
def apply_dynamic_operations(s: str, rules: List[Tuple[str, callable]]) -> Tuple[str, int]:
    """
    Apply the first applicable rule from the provided set of rules.
    Returns a tuple containing the modified string and the rule index (1-based within the selected rules)
    that was applied. If no rule is applicable, returns (s, 0).
    """
    for idx, (_, rule_fn) in enumerate(rules):
        new_s, op_idx = rule_fn(s)
        if op_idx > 0:
            return new_s, idx + 1  # Return 1-based index for the selected rules
    return s, 0

def transform_string_dynamic(s: str, selected_rules: List[Tuple[str, callable]]) -> List[str]:
    """
    Repeatedly apply the selected rules to the string until no further operations can be performed.
    Returns a list of string states from the initial string to the final string.
    """
    states = [s]
    while True:
        new_s, op_idx = apply_dynamic_operations(states[-1], selected_rules)
        if op_idx == 0:
            break
        states.append(new_s)
    return states

# ----------------------
# 3. Random String Generator Function
# ----------------------
def generate_random_string(min_len: int = 6, max_len: int = 12) -> str:
    """
    Generate a random string composed of characters 'a', 'b', 'c' with a length between min_len and max_len.
    """
    length = random.randint(min_len, max_len)
    return ''.join(random.choice(['a', 'b', 'c']) for _ in range(length))

# ----------------------
# 4. Dataset Generation Functions
# ----------------------
def compute_complexity(s: str, states: List[str]) -> int:
    """
    Define a custom 'complexity' metric based on the number of operations applied.
    Here, complexity = min(5, max(1, number_of_operations))
    """
    n_ops = len(states) - 1
    return min(5, max(1, n_ops))

def generate_dataset_with_random_rules(
    output_dir: str,
    num_samples: int = 100,
    min_len: int = 10,
    max_len: int = 20,
    num_rules: int = 6
):
    """
    Generate a dataset where each sample uses a random selection of rules from the RULES_POOL.
    Each sample includes:
      - A question text with the list of randomly selected rules.
      - A solution file containing the initial string, the selected rules, intermediate states, final string, and complexity.
    Only samples with a complexity greater than 2 (i.e. at least 3 operations applied) are kept.
    """
    os.makedirs(output_dir, exist_ok=True)

    collected_samples = 0
    sample_index = 0

    while collected_samples < num_samples:
        # Generate a random input string
        rand_str = generate_random_string(min_len, max_len)

        # Randomly select a subset of rules from the pool
        selected_rules = random.sample(RULES_POOL, num_rules)

        # Apply the dynamic transformations based on the selected rules
        states = transform_string_dynamic(rand_str, selected_rules)
        final_str = states[-1]

        # Compute custom complexity (number of operations applied)
        cpx = compute_complexity(rand_str, states)
        if cpx <= 3:  # Only use samples with complexity > 2 (i.e. at least 3 operations)
            continue

        # Prepare the question text with the selected rules description
        rules_description = "\n".join([f"{i + 1}. {desc}" for i, (desc, _) in enumerate(selected_rules)])
        question_text = (
            "Cycle through the following operations one at a time until no further operations can be performed:\n\n"
            f"{rules_description}\n\n"
            f"For the string \"{rand_str}\", what is the final string after performing all operations?\n\n"
            "Provide your answer as a string in the following format:\n\n"
            "<<< \"some_final_string\" >>>\n"
        )

        # Create a subdirectory for this sample and save the question and solution
        sample_dir = os.path.join(output_dir, f"sample_{sample_index}")
        os.makedirs(sample_dir, exist_ok=True)

        with open(os.path.join(sample_dir, "question.txt"), "w", encoding="utf-8") as fq:
            fq.write(question_text)

        with open(os.path.join(sample_dir, "solution.json"), "w", encoding="utf-8") as fs:
            json.dump(
                {
                    "initial_string": rand_str,
                    "selected_rules": [desc for desc, _ in selected_rules],
                    "intermediate_states": states,
                    "final_string": final_str,
                    "complexity": cpx
                },
                fs,
                indent=2,
                ensure_ascii=False
            )

        print(f"Generated sample_{sample_index} | initial='{rand_str}', final='{final_str}', complexity={cpx}")
        collected_samples += 1
        sample_index += 1

# ----------------------
# 5. Main Execution
# ----------------------
if __name__ == "__main__":
    # Generate 100 samples with a random selection of 6 rules (out of 20) per sample.
    output_dir = "../dataset_gather/string_deletion_and_modification"
    generate_dataset_with_random_rules(output_dir=output_dir, num_samples=200, min_len=10, max_len=20, num_rules=6)
