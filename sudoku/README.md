# Sudoku Meta-Learning Playground

This project rebuilds the workspace as a two-layer learning system:

- Layer 2 is born fresh each life and tries to solve Sudoku puzzles.
- Layer 2 is not given Sudoku rules or any user input. It only receives a binary success signal based on whether its completed board satisfies the puzzle.
- Layer 1 evolves the architecture and lifetime training hyperparameters for Layer 2 so that each new life can learn faster from scratch.

## No gradient descent

Nothing in this project learns by gradient descent or backpropagation. There is no loss function, no derivative and no autograd. Both layers learn by trial, selection and keeping what worked.

**Layer 2 (one life): perturb, test, keep the best.**

- The learner is a small feed-forward network written in plain Python, with no ML framework.
- Each step, the current weights are copied several times and each copy gets random Gaussian noise (`jitter_from`).
- Every noisy copy tries a batch puzzle and gets a binary reward: `1.0` if the board is a valid solution, `0.0` otherwise.
- The weights then move a fraction of the way toward the best noisy copy (`update_towards`, scaled by `learning_rate` and `elite_bias`).
- This is an evolution-strategy style update. It uses only the reward, never the slope of an error, so a sparse 0/1 signal is enough.

**Layer 1 (across lives): evolve the learner itself.**

- Each generation, the champion architecture is mutated (`mutate_architecture`): layer sizes, activation, learning rate, noise scale, temperature, steps and the clue curriculum.
- A new Layer 2 is born with fresh random weights and lives under that architecture.
- Architectures whose lives solve more puzzles are kept, so each new life can learn faster from scratch.

**Why this way.** Gradient descent needs a differentiable path from reward to weights. Here the reward is a hard pass or fail check on a finished board, so there is nothing to differentiate. The cost is sample efficiency: every update needs many full rollouts, which is why the clue curriculum and the outer search matter.

## Run

```bash
python server.py
```

Then open `http://127.0.0.1:8000`.

## Notes

- The reward is intentionally sparse: `1.0` only when the final board is a valid completion that matches the puzzle clues, otherwise `0.0`.
- To keep the search tractable, the environment uses a curriculum over clue counts. Layer 1 can evolve that curriculum too.
- The playground is display-only. There are no controls for editing the board.