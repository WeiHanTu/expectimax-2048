"""
This module implements an AI agent for the 2048 game using the Expectimax algorithm.

Classes:
    Node: Represents a node in the game tree.
    AI: Implements the AI agent to determine the next move using Expectimax.

Functions:
    run_experiment(depth, eval_function="score", num_moves=1000): Runs an experiment to evaluate the AI's performance.
    plot_comparison(runs=5, num_moves=1000): Plots a comparison of the AI's performance using different depths.
    plot_advanced_comparison(runs=5, num_moves=1000): Plots a comparison of the AI's performance using different evaluation functions.

Usage:
    To run the experiments and generate plots, execute this module as the main program.
"""

from __future__ import absolute_import, division, print_function
import copy, random
from game import Game
import numpy as np
import matplotlib.pyplot as plt

MOVES = {0: 'up', 1: 'left', 2: 'down', 3: 'right'}
MAX_PLAYER, CHANCE_PLAYER = 0, 1

# Tree node. To be used to construct a game tree.
class Node:
    """
    Represents a node in the game tree.

    Attributes:
        state (tuple): The state of the game at this node.
        player_type (int): The type of player (MAX_PLAYER or CHANCE_PLAYER).
        children (list): A list of (direction, node) tuples representing the children of this node.
    """
    def __init__(self, state, player_type):
        self.state = (state[0], state[1])
        # to store a list of (direction, node) tuples
        self.children = []
        self.player_type = player_type

    # returns whether this is a terminal state (i.e., no children)
    def is_terminal(self):
        """
        Checks if this node is a terminal state (i.e., no children).

        Returns:
            bool: True if this node is terminal, False otherwise.
        """
        return len(self.children) == 0

# AI agent. Determine the next move.
class AI:
    """
    AI agent to determine the next move using the Expectimax algorithm.

    Attributes:
        root (Node): The root node of the game tree.
        search_depth (int): The depth to which the game tree is built.
        simulator (Game): A simulator for the game.
        evaluation_function (callable): The evaluation function to use for leaf nodes.
    """
    def __init__(self, root_state, search_depth=3, evaluation_function=None):
        self.root = Node(root_state, MAX_PLAYER)
        self.search_depth = search_depth
        self.simulator = Game(*root_state)
        # If a custom evaluation function is provided, use it.
        if evaluation_function is not None and callable(evaluation_function):
            self.evaluation_function = evaluation_function

    def build_tree(self, node = None, depth = 0):
        """
        Builds the game tree up to the specified depth.

        Args:
            node (Node): The current node in the game tree.
            depth (int): The remaining depth to build.
        """
        if node is None:
            node = self.root
        if depth == 0:
            return
        if node.player_type == MAX_PLAYER:
            for move in range(4):
                sim = copy.deepcopy(self.simulator)
                sim.set_state(*node.state)
                # Only expand if move is feasible.
                if sim.move(move):
                    child_state = sim.current_state()
                    child_node = Node(child_state, CHANCE_PLAYER)
                    node.children.append((move, child_node))
                    self.build_tree(child_node, depth - 1)
                else:
                    # For infeasible moves, annotate as terminal with a very negative score.
                    child_state = (node.state[0], -100)
                    child_node = Node(child_state, CHANCE_PLAYER)
                    node.children.append((move, child_node))
                    # Do not expand further for terminal nodes.
        elif node.player_type == CHANCE_PLAYER:
            sim = copy.deepcopy(self.simulator)
            sim.set_state(*node.state)
            open_tiles = sim.get_open_tiles()
            if not open_tiles:
                return
            # Iterate over all open tiles rather than sampling.
            for tile in open_tiles:
                sim_copy = copy.deepcopy(sim)
                sim_copy.tile_matrix[tile[0]][tile[1]] = 2  # Place a new tile.
                child_state = sim_copy.current_state()
                child_node = Node(child_state, MAX_PLAYER)
                node.children.append((None, child_node))
                self.build_tree(child_node, depth - 1)
    def expectimax(self, node=None):
        """
        Performs the Expectimax algorithm to evaluate the game tree.

        Args:
            node (Node): The current node in the game tree.

        Returns:
            tuple: The best move and its value.
        """
        if node is None:
            node = self.root
        if node.is_terminal():
            return (None, self.evaluation_function(node.state))
        if node.player_type == MAX_PLAYER:
            best_val = -float('inf')
            best_move = None
            for action, child in node.children:
                _, val = self.expectimax(child)
                if val > best_val:
                    best_val = val
                    best_move = action
            return (best_move, best_val)
        else:
            total = 0
            for _, child in node.children:
                _, val = self.expectimax(child)
                total += val
            avg_val = total / (len(node.children) if node.children else 1)
            return (None, avg_val)


    def evaluation_function(self, state):
        """
        Default evaluation function that uses the game score as the payoff value.

        Args:
            state (tuple): The state of the game.

        Returns:
            int: The score of the game.
        """
        board, score = state
        return score  # Use the game score as the payoff value.

    # Return decision at the root  (Exp-3)
    def compute_decision(self):
        """
        Computes the best move using the Expectimax algorithm.

        Returns:
            int: The best move.
        """
        self.build_tree(self.root, self.search_depth)
        direction, _ = self.expectimax(self.root)
        return direction

    # Return decision at the root (Exp-1)
    def compute_decision_exp1(self):
        """
        Computes the best move using a depth-1 search.

        Returns:
            int: The best move.
        """
        best_move = None
        best_value = -float('inf')
        for move in range(4):
            sim = copy.deepcopy(self.simulator)
            sim.set_state(*self.root.state)
            if sim.move(move):
                state = sim.current_state()
            else:
                board, _ = self.root.state
                state = (board, -100)
            value = self.evaluation_function(state)
            if value > best_value:
                best_value = value
                best_move = move
        return best_move

    def evaluation_function_custom(self, state):
        """
        Advanced evaluation function that considers multiple strategic factors:
        1. Current score
        2. Number of empty cells (mobility)
        3. Monotonicity (tiles should be ordered)
        4. Smoothness (adjacent tiles should be similar)
        5. Maximum tile placement (higher values in corners)
        6. Merge potential (adjacent tiles with same values)

        Args:
            state (tuple): The current state of the game, consisting of the board and the score.

        Returns:
            float: The heuristic value of the state based on the combined strategic factors.
        """
        board, score = state

        # Get empty cells (mobility)
        empty_cells = sum(row.count(0) for row in board)
        empty_weight = 2.7 * empty_cells

        # Monotonicity: Prefer ordered tiles
        monotonicity_score = 0
        for i in range(4):
            # Check rows
            row = board[i]
            if row[0] >= row[1] >= row[2] >= row[3]:
                monotonicity_score += sum(row)
            # Check columns
            col = [board[j][i] for j in range(4)]
            if col[0] >= col[1] >= col[2] >= col[3]:
                monotonicity_score += sum(col)

        # Smoothness: Prefer similar values next to each other
        smoothness = 0
        for i in range(4):
            for j in range(4):
                if board[i][j] != 0:
                    # Check horizontal neighbor
                    if j < 3 and board[i][j + 1] != 0:
                        smoothness -= abs(board[i][j] - board[i][j + 1])
                    # Check vertical neighbor
                    if i < 3 and board[i + 1][j] != 0:
                        smoothness -= abs(board[i][j] - board[i + 1][j])

        # Corner preference: Highest values should be in corners
        corner_score = 0
        corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
        max_tile = max(max(row) for row in board)
        for i, j in corners:
            if board[i][j] == max_tile:
                corner_score = max_tile * 2

        # Merge potential: Adjacent same values
        merge_score = 0
        for i in range(4):
            for j in range(4):
                if board[i][j] != 0:
                    # Check horizontal merge
                    if j < 3 and board[i][j] == board[i][j + 1]:
                        merge_score += board[i][j] * 1.3
                    # Check vertical merge
                    if i < 3 and board[i][j] == board[i + 1][j]:
                        merge_score += board[i][j] * 1.3

        # Weight the different components
        total_score = (
                score +
                empty_weight * 0.3 +
                monotonicity_score * 0.5 +
                smoothness * 0.1 +
                corner_score * 1.2 +
                merge_score * 1.1
        )

        return total_score

    def compute_decision_ec(self):
        """
        Computes the best move using the Expectimax algorithm with a custom evaluation function.
        The search depth is adjusted based on the number of empty cells in the current state.

        Returns:
            int: The best move direction.
        """
        # Get number of empty cells to adjust search depth
        empty_cells = sum(row.count(0) for row in self.root.state[0])

        # Adjust search depth based on number of empty cells
        if empty_cells > 8:
            self.search_depth = 4  # More empty cells -> can look deeper
        elif empty_cells > 4:
            self.search_depth = 3  # Medium number of cells -> standard depth
        else:
            self.search_depth = 2  # Few empty cells -> need to be faster

        original_eval = self.evaluation_function
        self.evaluation_function = self.evaluation_function_custom
        self.build_tree(self.root, self.search_depth)
        direction, _ = self.expectimax(self.root)
        self.evaluation_function = original_eval
        return direction


# ----- Experimentation and Plotting -----

def run_experiment(depth, eval_function="score", num_moves=1000):
    """
    Runs an experiment to evaluate the AI's performance.

    Args:
        depth (int): The search depth for the AI.
        eval_function (str): The evaluation function to use ("score" or "advanced").
        num_moves (int): The number of moves to simulate.

    Returns:
        list: A list of scores after each move.
    """
    scores = []
    game = Game()
    # We pass eval_function as is; later we decide which decision method to call.
    ai = AI(game.current_state(), depth, None)

    for _ in range(num_moves):
        if game.game_over():
            break

        # Choose decision method based on depth and evaluation type.
        if depth == 1:
            direction = ai.compute_decision_exp1()
        elif depth == 3:
            if eval_function == "advanced":
                direction = ai.compute_decision_ec()
            else:
                direction = ai.compute_decision()
        else:
            direction = ai.compute_decision()

        game.move_and_place(direction)
        scores.append(game.score)
        # Reset AI with new game state.
        ai = AI(game.current_state(), depth, None)
    return scores


def plot_comparison(runs=5, num_moves=1000):
    """
    Plots a comparison of the AI's performance using different depths.

    Args:
        runs (int): The number of runs to perform.
        num_moves (int): The number of moves to simulate in each run.
    """
    plt.figure(figsize=(12, 6))
    # Plot Exp-1 performance.
    for i in range(runs):
        scores = run_experiment(1, "score", num_moves)
        # print(f"Exp-1 Run {i + 1},  Score: {scores}")
        plt.plot(scores, alpha=0.5, label=f'Exp-1 Run {i + 1}')
    # Plot Exp-3 performance (default evaluation).
    for i in range(runs):
        scores = run_experiment(3, "score", num_moves)
        # print(f"Exp-3 Run {i + 1},  Score: {scores}")
        plt.plot(scores, alpha=0.5, label=f'Exp-3 Run {i + 1}')
    plt.xlabel('Number of Moves')
    plt.ylabel('Game Score')
    plt.title('Performance Comparison: Exp-1 vs Exp-3')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_advanced_comparison(runs=5, num_moves=1000):
    """
    Plots a comparison of the AI's performance using different evaluation functions.

    Args:
        runs (int): The number of runs to perform.
        num_moves (int): The number of moves to simulate in each run.
    """
    plt.figure(figsize=(12, 6))
    # Plot original Exp-3.
    for i in range(runs):
        scores = run_experiment(3, "score", num_moves)
        # print(f"Original Exp-3 Run {i + 1},  Score: {scores}")
        plt.plot(scores, alpha=0.5, label=f'Original Exp-3 Run {i + 1}')
    # Plot advanced Exp-3 (custom evaluation).
    for i in range(runs):
        scores = run_experiment(3, "advanced", num_moves)
        # print(f"Advanced Exp-3 Run {i + 1},  Score: {scores}")
        plt.plot(scores, alpha=0.5, label=f'Advanced Exp-3 Run {i + 1}')
    plt.xlabel('Number of Moves')
    plt.ylabel('Game Score')
    plt.title('Performance Comparison: Original vs Advanced Evaluation')
    plt.legend()
    plt.grid(True)
    plt.show()


# Run experiments and generate plots
if __name__ == "__main__":
    print("Running experiments...")
    plot_comparison()
    plot_advanced_comparison()
    print("Experiments completed.")
