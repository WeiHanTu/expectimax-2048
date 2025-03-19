# Expectimax Search for 2048

This project implements an AI agent for the 2048 game using the Expectimax algorithm. The AI agent is designed to make optimal moves by evaluating possible future game states.

## Features

- **AI Agent**: Uses the Expectimax algorithm to determine the best move.
- **Game Engine**: Handles the game mechanics, including tile movements, merging, and random tile placement.
- **User Interface**: Allows manual play using arrow keys and toggles for AI play.
- **Testing**: Includes tests to verify the AI's performance and correctness.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/WeiHanTu/expectimax-main.git
    cd expectimax-main
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

Usage
-----
To run the program:
```
    python main.py
```

## Keyboard Controls
- Arrow Keys: Move tiles manually.
- Enter: Toggle AI play.
- 'r': restart the game
- 'u': undo a move
- '3'-'7': change board size
- 'g': toggle grayscale
- 'e': switch to extra credit

## Running Tests
To run the tests:
```
    python main.py -t 1
```

## AI Implementation
The AI agent is implemented using the Expectimax algorithm. The AI evaluates possible future game states up to a specified depth and chooses the move that maximizes the expected score.  
AI Classes and Functions

## AI Classes and Functions
- Node: Represents a node in the game tree. 
- AI: Implements the AI agent to determine the next move using Expectimax. 
- run_experiment: Runs an experiment to evaluate the AI's performance. 
- plot_comparison: Plots a comparison of the AI's performance using different depths. 
- plot_advanced_comparison: Plots a comparison of the AI's performance using different evaluation functions.

## Extra Credit
The extra credit mode uses a custom evaluation function that considers multiple strategic factors, such as the number of empty cells, monotonicity, smoothness, corner preference, and merge potential.

## License
This project is licensed under the MIT License.
