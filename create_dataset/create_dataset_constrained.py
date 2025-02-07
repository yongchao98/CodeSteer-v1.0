import random
import os
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import re

class GameType(Enum):
    ELEMENTS = "elements"
    ANIMAL = "animal"
    STICK_GAME = "stick_game"

@dataclass
class GameRound:
    player_move: str
    result: str  # "win", "loss", "draw"

@dataclass
class GamePuzzle:
    game_type: GameType
    rules: str
    rounds: List[GameRound]
    solution: List[str]
    complexity: int  # 1-5 scale

class GameRulesets:
    @staticmethod
    def get_element_rules() -> Dict:
        return {
            "pieces": ["A", "B", "C", "D", "E"],  # Metal, Wood, Water, Fire, Earth
            "generates": {"A": "B", "B": "C", "C": "D", "D": "E", "E": "A"},
            "overcomes": {"A": "C", "C": "E", "E": "B", "B": "D", "D": "A"}
        }
    
    @staticmethod
    def get_stick_game_rules() -> Dict:
        return {
            "pieces": ["A", "B", "C", "D"],  # Stick, Tiger, Chicken, Worm
            "beats": {"A": "B", "B": "C", "C": "D", "D": "A"}
        }
    
    @staticmethod
    def get_animal_rules() -> Dict:
        return {
            "pieces": ["A", "B", "C", "D", "E", "F", "G", "H"],  # Elephant to Mouse
            "special": {"H": "A"},  # Mouse beats Elephant
            "hierarchy": {"A": 8, "B": 7, "C": 6, "D": 5, "E": 4, "F": 3, "G": 2, "H": 1}
        }

class GamePuzzleGenerator:
    def __init__(self):
        self.rulesets = {
            GameType.ELEMENTS: GameRulesets.get_element_rules(),
            GameType.STICK_GAME: GameRulesets.get_stick_game_rules(),
            GameType.ANIMAL: GameRulesets.get_animal_rules()
        }

    def determine_winner_elements(self, player: str, opponent: str) -> str:
        rules = self.rulesets[GameType.ELEMENTS]
        if rules["generates"].get(player) == opponent:
            return "loss"
        elif rules["overcomes"].get(player) == opponent:
            return "win"
        return "draw"

    def determine_winner_stick(self, player: str, opponent: str) -> str:
        rules = self.rulesets[GameType.STICK_GAME]
        if rules["beats"].get(player) == opponent:
            return "win"
        elif rules["beats"].get(opponent) == player:
            return "loss"
        return "draw"

    def determine_winner_animal(self, player: str, opponent: str) -> str:
        rules = self.rulesets[GameType.ANIMAL]
        if player == opponent:
            return "draw"
        if player == "H" and opponent == "A":  # Mouse beats Elephant
            return "win"
        if opponent == "H" and player == "A":  # Mouse beats Elephant
            return "loss"
        return "win" if rules["hierarchy"][player] > rules["hierarchy"][opponent] else "loss"

    def generate_round_moves(self, pieces: List[str], num_rounds: int) -> Tuple[List[str], List[str]]:
        """Generate moves for both players ensuring each piece is used only once per player"""
        # Create separate piece pools for each player
        player_pieces = pieces.copy()
        opponent_pieces = pieces.copy()
        
        player_moves = []
        opponent_moves = []
        
        for _ in range(num_rounds):
            if not player_pieces or not opponent_pieces:
                raise ValueError(f"Not enough pieces for {num_rounds} rounds")
                
            # Select moves for both players
            player_move = random.choice(player_pieces)
            player_pieces.remove(player_move)
            player_moves.append(player_move)
            
            opponent_move = random.choice(opponent_pieces)
            opponent_pieces.remove(opponent_move)
            opponent_moves.append(opponent_move)
            
        return player_moves, opponent_moves

    def generate_puzzle(self, game_type: GameType, num_rounds: int, complexity: int) -> GamePuzzle:
        rules = self.rulesets[game_type]
        pieces = rules["pieces"]
        
        # Validate number of rounds
        if num_rounds > len(pieces):
            raise ValueError(f"Number of rounds ({num_rounds}) cannot exceed number of pieces ({len(pieces)})")
        
        # Generate moves for both players
        try:
            player_moves, opponent_moves = self.generate_round_moves(pieces, num_rounds)
        except ValueError as e:
            raise ValueError(f"Failed to generate puzzle: {str(e)}")
        
        rounds = []
        results = []  # Store results for each round
        for i in range(num_rounds):
            player_piece = player_moves[i]
            opponent_piece = opponent_moves[i]
            
            if game_type == GameType.ELEMENTS:
                result = self.determine_winner_elements(player_piece, opponent_piece)
            elif game_type == GameType.STICK_GAME:
                result = self.determine_winner_stick(player_piece, opponent_piece)
            else:
                result = self.determine_winner_animal(player_piece, opponent_piece)
                
            rounds.append(GameRound(player_piece, result))
            results.append(result)  # Add result to results list

        rules_text = self.generate_rules_text(game_type)
        # Add results to the puzzle object
        puzzle = GamePuzzle(game_type, rules_text, rounds, opponent_moves, complexity)
        # Return puzzle, results, and player moves
        return puzzle, results, player_moves
    def generate_rules_text(self, game_type: GameType) -> str:
        if game_type == GameType.ELEMENTS:
            return """Both parties, A and B, each have a set of five-element flags, with each person holding five cards representing Metal, Wood, Water, Fire, and Earth. Each card can only be used once. Both players play their cards simultaneously without seeing the opponent's card, and the referee announces the result of each round.
Rules:
1. When card 1 generates card 2, the holder of card 1 loses;
2. When card 1 overcomes card 2, the holder of card 1 wins;
3. When both cards are the same or no card generates or overcomes each other, it is a draw.
Known:
1. Generation: Wood A generates Fire B, Fire B generates Earth C, Earth C generates Metal D, Metal D generates Water E, Water E generates Wood A.
2. Overcoming: Wood A overcomes Earth C, Earth C overcomes Water E, Water E overcomes Fire B, Fire B overcomes Metal D, Metal D overcomes Wood A."""
        elif game_type == GameType.STICK_GAME:
            return """Both parties, A and B, are playing a game called "Stick, Tiger, Chicken." Each person has four cards: Stick (A), Tiger (B), Chicken (C), and Worm (D). Each card can only be used once. Both players play their cards simultaneously without seeing the other's card, and a referee announces the result of each round.

Rules:
1. Stick beats Tiger, Tiger eats Chicken, Chicken eats Worm, Worm beats Stick, and so on in a cycle.
2. If both players play the same card or cards that are one step apart in the cycle, it is considered a draw."""
        else:
            return """Both parties, A and B, play a game similar to Jungle Chess. Each player has 8 animal pieces, ranked from largest to smallest: Elephant (A), Lion (B), Tiger (C), Leopard (D), Wolf (E), Dog (F), Cat (G), and Mouse (H). Both players reveal their pieces simultaneously, but cannot see the opponent's piece. The referee announces the result of the match.

Rules:
1. The goal for both players is to reach the opponent's den in 5 steps. The first to accumulate 5 steps wins.
2. When a smaller piece encounters a larger piece, the smaller piece is captured, and the player with the larger piece wins.
3. Mutual capture: When pieces of the same type encounter each other, they mutually capture and the result is a draw.
4. Mouse captures Elephant: Apart from the size hierarchy, the Mouse can capture the Elephant, but the Elephant cannot capture the Mouse.
5. Each piece can only be used once."""

def check_answer(answer: str, solution: List[str]) -> bool:
    """Check if the provided answer matches the solution"""
    # Extract content between <<< and >>>
    match = re.search(r'<<<(.+?)>>>', answer)
    if not match:
        return False
    
    # Parse the content into a list
    try:
        content = match.group(1)
        # Remove any whitespace and split by commas
        parsed_answer = [x.strip() for x in content.strip('[]').split(',')]
        # Compare with solution
        return parsed_answer == solution
    except:
        return False

def generate_dataset(output_dir: str):
    """Generate a dataset of game puzzles with varying complexity"""
    os.makedirs(output_dir, exist_ok=True)
    generator = GamePuzzleGenerator()
    
    configurations = [
        # (game_type, num_rounds, complexity, num_samples)
        (GameType.ELEMENTS, 3, 3, 2),
        (GameType.STICK_GAME, 4, 3, 2),
        (GameType.ANIMAL, 3, 3, 2),
        (GameType.ELEMENTS, 3, 4, 2),
        (GameType.STICK_GAME, 4, 4, 2),
        (GameType.ANIMAL, 3, 4, 2),
        (GameType.ELEMENTS, 3, 5, 2),
        (GameType.STICK_GAME, 4, 5, 24),
        (GameType.ANIMAL, 3, 5, 2),
        (GameType.ELEMENTS, 4, 5, 14),
        (GameType.ANIMAL, 4, 5, 16),
        (GameType.ELEMENTS, 5, 5, 14),
        (GameType.ANIMAL, 5, 5, 16),
        (GameType.ELEMENTS, 3, 3, 2),
        (GameType.STICK_GAME, 4, 3, 2),
        (GameType.ANIMAL, 3, 3, 2),
        (GameType.ELEMENTS, 3, 4, 2),
        (GameType.STICK_GAME, 4, 4, 2),
        (GameType.ANIMAL, 3, 4, 2),
        (GameType.ELEMENTS, 3, 5, 2),
        (GameType.STICK_GAME, 4, 5, 24),
        (GameType.ANIMAL, 3, 5, 2),
        (GameType.ELEMENTS, 4, 5, 14),
        (GameType.ANIMAL, 4, 5, 16),
        (GameType.ELEMENTS, 5, 5, 14),
        (GameType.ANIMAL, 5, 5, 16),
    ]
    
    sample_id = 0
    for game_type, num_rounds, complexity, num_samples in configurations:
        for _ in range(num_samples):
            try:
                puzzle, results, player_moves = generator.generate_puzzle(game_type, num_rounds, complexity)
                
                sample_dir = os.path.join(output_dir, f'sample_{sample_id}')
                os.makedirs(sample_dir, exist_ok=True)
                
                # Format question
                question = (
                    f"{puzzle.rules}\n\n"
                    "Assuming you are one of the players, deduce what cards your opponent played each round based on the cards you played and the results of each round.\n\n"
                    "Rounds:\n"
                )
                for i, round_info in enumerate(puzzle.rounds, 1):
                    question += f"Round {i}: {round_info.player_move}, {round_info.result}.\n"
                
                question += "\nUse a list of letters to show your answer. For example, <<<A,B,C>>>."
                
                # Save question and solution
                with open(os.path.join(sample_dir, 'question.txt'), 'w') as f:
                    f.write(question)
                    
                with open(os.path.join(sample_dir, 'solution.json'), 'w') as f:
                    json.dump({
                        'game_type': game_type.value,
                        'player_moves': player_moves,     # Add player moves
                        'solution': puzzle.solution,  # Renamed for clarity
                        'results': results,
                        'complexity': puzzle.complexity
                    }, f, indent=2)
                    
                sample_id += 1
                print(f"Generated sample {sample_id}: Player moves: {player_moves}, Opponent moves: {puzzle.solution}, Results: {results}")
            except ValueError as e:
                print(f"Skipping invalid configuration: {str(e)}")
                continue


if __name__ == "__main__":
    output_dir = '../dataset_gather/constrained_linear_arrangement'
    generate_dataset(output_dir)