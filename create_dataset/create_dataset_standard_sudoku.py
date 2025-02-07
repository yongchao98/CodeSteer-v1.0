import random
import os
import json
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SudokuPuzzle:
    puzzle: List[List[int]]
    solution: List[List[int]]
    complexity: int


def find_empty_cell(board: List[List[int]]) -> Optional[Tuple[int, int]]:
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return (r, c)
    return None


def is_valid(board: List[List[int]], row: int, col: int, val: int) -> bool:
    if val in board[row]:
        return False
    for r in range(9):
        if board[r][col] == val:
            return False
    subgrid_row = (row // 3) * 3
    subgrid_col = (col // 3) * 3
    for r in range(subgrid_row, subgrid_row + 3):
        for c in range(subgrid_col, subgrid_col + 3):
            if board[r][c] == val:
                return False
    return True


def solve_sudoku(board: List[List[int]]) -> bool:
    find_result = find_empty_cell(board)
    if not find_result:
        return True
    row, col = find_result

    for val in range(1, 10):
        if is_valid(board, row, col, val):
            board[row][col] = val
            if solve_sudoku(board):
                return True
            board[row][col] = 0
    return False


def fill_3x3_box(board: List[List[int]], row_start: int, col_start: int):
    digits = list(range(1, 10))
    random.shuffle(digits)
    for r in range(3):
        for c in range(3):
            board[row_start + r][col_start + c] = digits.pop()


def copy_board(board: List[List[int]]) -> List[List[int]]:
    return [row[:] for row in board]


def generate_full_solution() -> List[List[int]]:
    board = [[0] * 9 for _ in range(9)]
    for i in range(0, 9, 3):
        fill_3x3_box(board, i, i)
    solve_sudoku(board)
    return board


def count_solutions(board: List[List[int]], limit: int) -> int:
    find_result = find_empty_cell(board)
    if not find_result:
        return 1
    row, col = find_result
    count = 0
    for val in range(1, 10):
        if is_valid(board, row, col, val):
            board[row][col] = val
            count += count_solutions(board, limit)
            board[row][col] = 0
            if count >= limit:
                break
    return count


def has_unique_solution(board: List[List[int]]) -> bool:
    board_copy = copy_board(board)
    return count_solutions(board_copy, limit=2) == 1


def generate_puzzle(complexity: int) -> SudokuPuzzle:
    solution_board = generate_full_solution()
    puzzle_board = copy_board(solution_board)

    if complexity == 1:
        cells_to_remove = random.randint(30, 35)
    elif complexity == 2:
        cells_to_remove = random.randint(36, 40)
    elif complexity == 3:
        cells_to_remove = random.randint(41, 45)
    elif complexity == 4:
        cells_to_remove = random.randint(46, 50)
    else:
        cells_to_remove = random.randint(51, 55)

    all_cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(all_cells)

    removed = 0
    for (r, c) in all_cells:
        if removed >= cells_to_remove:
            break
        saved = puzzle_board[r][c]
        puzzle_board[r][c] = 0
        if not has_unique_solution(puzzle_board):
            puzzle_board[r][c] = saved
        else:
            removed += 1

    return SudokuPuzzle(
        puzzle=puzzle_board,
        solution=solution_board,
        complexity=complexity
    )


def puzzle_to_string(puzzle: List[List[int]]) -> str:
    rows_str = []
    for row in puzzle:
        row_str = []
        for val in row:
            row_str.append(str(val) if val != 0 else ' ')
        rows_str.append(','.join(row_str))
    return '\n'.join(rows_str)


def generate_dataset_sudoku(output_dir: str, num_samples_per_complexity: int = 10):
    os.makedirs(output_dir, exist_ok=True)

    sample_id = 0
    for complexity in range(5, 8):
        for _ in range(num_samples_per_complexity):
            puzzle_obj = generate_puzzle(complexity)

            sample_dir = os.path.join(output_dir, f'sample_{sample_id}')
            os.makedirs(sample_dir, exist_ok=True)

            puzzle_str = puzzle_to_string(puzzle_obj.puzzle)
            question = (
                "Given a 9*9 Sudoku grid, each cell must be filled with a number between 1 and 9. "
                "No number may appear more than once in the same row, column, or 3*3 subgrid. "
                "Some cells are pre-filled. Please fill the remaining empty cells.\n"
                f"{puzzle_str}\n\n"
                "Return your answer in the format: <<<a 9*9 matrix>>>."
            )
            with open(os.path.join(sample_dir, 'question.txt'), 'w') as f:
                f.write(question)

            with open(os.path.join(sample_dir, 'solution.json'), 'w') as f:
                json.dump({
                    "puzzle": puzzle_obj.puzzle,  # Included question matrix
                    "solution": puzzle_obj.solution,
                    "complexity": puzzle_obj.complexity
                }, f, indent=2)

            sample_id += 1
            print(f"Generated sample {sample_id} with complexity {complexity}")


def parse_sudoku_matrix(llm_answer: str) -> List[List[int]]:
    inside_brackets = re.findall(r'<<<(.*?)>>>', llm_answer, flags=re.DOTALL)
    if inside_brackets:
        text_to_parse = inside_brackets[0].strip()
    else:
        text_to_parse = llm_answer

    lines = [ln.strip() for ln in text_to_parse.splitlines() if ln.strip()]
    if len(lines) < 9:
        raise ValueError("Not enough lines for a 9x9 matrix.")

    matrix = []
    for line in lines[:9]:
        parts = re.split(r'[,\s]+', line.strip())
        parts = [p for p in parts if p]
        if len(parts) != 9:
            raise ValueError(f"Line does not contain 9 numbers: '{line}'")
        try:
            row = [int(x) for x in parts]
        except ValueError:
            raise ValueError(f"Non-integer value found in row: '{line}'")
        matrix.append(row)
    if len(matrix) != 9 or any(len(row) != 9 for row in matrix):
        raise ValueError("Parsed matrix is not 9x9.")
    return matrix


def check_sudoku_solution(
        puzzle: List[List[int]],
        solution_candidate: List[List[int]]
) -> bool:
    # Check givens
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] != 0 and puzzle[r][c] != solution_candidate[r][c]:
                return False
    # Check rows
    for r in range(9):
        row_vals = set()
        for c in range(9):
            val = solution_candidate[r][c]
            if val < 1 or val > 9 or val in row_vals:
                return False
            row_vals.add(val)
    # Check columns
    for c in range(9):
        col_vals = set()
        for r in range(9):
            val = solution_candidate[r][c]
            if val < 1 or val > 9 or val in col_vals:
                return False
            col_vals.add(val)
    # Check 3x3 subgrids
    for sub_row in range(0, 9, 3):
        for sub_col in range(0, 9, 3):
            box_vals = set()
            for r in range(sub_row, sub_row + 3):
                for c in range(sub_col, sub_col + 3):
                    val = solution_candidate[r][c]
                    if val in box_vals:
                        return False
                    box_vals.add(val)
    return True


def check_llm_sudoku_answer(puzzle: List[List[int]], llm_answer: str) -> bool:
    try:
        candidate_matrix = parse_sudoku_matrix(llm_answer)
        return check_sudoku_solution(puzzle, candidate_matrix)
    except Exception as e:
        print(f"Parsing/validation error: {e}")
        return False


if __name__ == "__main__":
    # Define the output directory for the dataset
    output_dir = '../dataset_gather/standard_sudoku'

    # Generate the dataset with 10 samples per complexity level
    generate_dataset_sudoku(output_dir, num_samples_per_complexity=40)

    # Example of checking an LLM's answer
    # Assume you have an LLM's answer as a string
    llm_answer_text = """
    Here's my solution:
    <<<a 9*9 matrix>>>
    5,3,1,9,2,6,7,8,5
    4,8,7,5,1,3,9,2,6
    6,2,9,7,4,8,3,1,9
    8,4,6,2,9,5,3,1,4
    2,3,5,4,8,7,6,9,1
    9,2,1,8,3,5,4,4,7
    7,5,3,6,1,9,2,3,4
    3,6,9,1,7,1,2,8,5
    1,7,4,3,5,2,8,7,2
    <<<end>>>
    """

    # Example puzzle (replace with actual puzzle from your dataset)
    example_puzzle = [
        [0, 0, 0, 9, 2, 0, 7, 0, 5],
        [0, 0, 7, 5, 1, 0, 0, 2, 0],
        [6, 0, 0, 7, 4, 0, 3, 0, 9],
        [8, 4, 0, 2, 0, 5, 3, 1, 0],
        [0, 3, 5, 0, 8, 0, 0, 9, 0],
        [9, 2, 0, 0, 0, 5, 0, 4, 7],
        [0, 0, 3, 0, 0, 9, 0, 0, 4],
        [0, 6, 9, 0, 7, 1, 2, 8, 0],
        [1, 0, 4, 3, 5, 2, 0, 7, 0],
    ]

    # Check if the LLM's answer is correct
    is_correct = check_llm_sudoku_answer(example_puzzle, llm_answer_text)
    print("LLM's solution is correct!" if is_correct else "LLM's solution is incorrect.")
