# Sudoku Meta-Learning Playground

This project rebuilds the workspace as a two-layer learning system:

- Layer 2 is born fresh each life and tries to solve Sudoku puzzles.
- Layer 2 is not given Sudoku rules or any user input. It only receives a binary success signal based on whether its completed board satisfies the puzzle.
- Layer 1 evolves the architecture and lifetime training hyperparameters for Layer 2 so that each new life can learn faster from scratch.

## Run

```bash
python server.py
```

Then open `http://127.0.0.1:8000`.

## Notes

- The reward is intentionally sparse: `1.0` only when the final board is a valid completion that matches the puzzle clues, otherwise `0.0`.
- To keep the search tractable, the environment uses a curriculum over clue counts. Layer 1 can evolve that curriculum too.
- The playground is display-only. There are no controls for editing the board.