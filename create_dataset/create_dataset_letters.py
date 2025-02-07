import random
import re
import requests
import json
import string
import os

def download_word_list():
    """
    Download a list of English words from an online source.
    """
    url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text.splitlines()
    else:
        raise Exception("Failed to download word list")


# Download the word list (you may want to save this locally for repeated use)
WORD_LIST = download_word_list()


def synthesize_word(target_letter, count, min_length, max_length):
    """
    Synthesize a word with a specific count of a target letter.

    :param target_letter: The letter to be counted
    :param count: The number of times the target letter should appear
    :param min_length: The minimum length of the word
    :param max_length: The maximum length of the word
    :return: A synthesized word
    """
    length = random.randint(min_length, max_length)
    other_letters = [c for c in string.ascii_lowercase if c != target_letter]

    # Start with the required number of target letters
    word = [target_letter] * count

    # Fill the rest with random letters
    word.extend(random.choices(other_letters, k=length - count))

    # Shuffle the word
    random.shuffle(word)

    return ''.join(word)

def generate_test_words(target_letter, count, n, min_length=5, max_length=15):
    """
    Generate a list of test words with a specific count of a target letter.
    If there aren't enough real words, synthesize additional words.

    :param target_letter: The letter to be counted
    :param count: The number of times the target letter should appear
    :param n: The number of words to generate
    :param min_length: The minimum length of the word (default 5)
    :param max_length: The maximum length of the word (default 15)
    :return: A list of generated words
    """
    suitable_words = [word for word in WORD_LIST
                      if min_length <= len(word) <= max_length
                      and word.count(target_letter) == count]

    # If we have enough words, return a random sample
    if len(suitable_words) >= n:
        return random.sample(suitable_words, n)

    # If we don't have enough words, use all available words and synthesize the rest
    result = suitable_words.copy()
    num_to_synthesize = n - len(suitable_words)

    for _ in range(num_to_synthesize):
        synthesized_word = synthesize_word(target_letter, count, min_length, max_length)
        result.append(synthesized_word)

    # Shuffle the result to mix real and synthesized words
    random.shuffle(result)

    return result


def create_prompt(word, target_letter):
    """
    Create a prompt for the LLM based on the test word and target letter.

    :param word: The test word
    :param target_letter: The letter to be counted
    :return: A formatted prompt string
    """
    return f"How many '{target_letter}'s are in the word '{word}' and what are their positions? Please respond in the format: 'Count: X, Positions: [Y, Z, ...]'"


def evaluate_response(word, target_letter, llm_response):
    """
    Evaluate the LLM's response for correctness.

    :param word: The test word
    :param target_letter: The letter that was counted
    :param llm_response: The response from the LLM
    :return: A tuple (is_correct, explanation)
    """
    # Extract count and positions from LLM response
    match = re.search(r"Count: (\d+), Positions: \[([\d, ]+)\]", llm_response)
    if not match:
        return False, "Response format is incorrect"

    llm_count = int(match.group(1))
    llm_positions = [int(pos) for pos in match.group(2).split(',')]

    # Calculate correct count and positions
    correct_count = word.count(target_letter)
    correct_positions = [i + 1 for i, letter in enumerate(word) if letter == target_letter]

    if llm_count != correct_count:
        return False, f"Incorrect count. Expected {correct_count}, got {llm_count}"

    if set(llm_positions) != set(correct_positions):
        return False, f"Incorrect positions. Expected {correct_positions}, got {llm_positions}"

    return True, "Correct response"


def write_words_to_file(words, filename):
    """
    Write a list of words to a JSON file.

    :param words: List of words to write
    :param filename: Name of the file to write to
    """
    with open(filename, 'w') as f:
        json.dump(words, f)
    #print(f"Words written to {filename}")


def read_words_from_file(filename):
    """
    Read a list of words from a JSON file.

    :param filename: Name of the file to read from
    :return: List of words
    """
    with open(filename, 'r') as f:
        words = json.load(f)
    print(f"Words read from {filename}")
    return words


# Example usage
if __name__ == "__main__":
    for min_length, max_length in [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]:
        base_dir = f'/Users/yongchaochen/Robot_NLP/MSR_intern/code_text_decision/dataset_gather/Letters_dataset_min_length_{min_length}_max_length_{max_length}/'
        for letter in string.ascii_lowercase:
            for letter_freq in range(1, 6):
                words = generate_test_words(letter, letter_freq, 10, min_length=min_length, max_length=max_length)
                print(f"Min_length: {min_length}, Max_length: {max_length}, Letter: {letter}, Letter_freq: {letter_freq}, Generated words: {words}")
                for index in range(len(words)):
                    saving_dir = base_dir + f"{letter}_{letter_freq}_{index}/"
                    if not os.path.exists(saving_dir):
                        os.makedirs(saving_dir)
                    write_words_to_file(words[index], saving_dir + 'test_words.json')