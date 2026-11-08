# nullmind

Can a neural network work out a hidden Sudoku rule while its weights stay frozen, and be trained with no gradient descent at all?

**Status: plan only. No network has been trained yet.** The world numbers below come from small simulations of scripted agents (1000 trials per setting). Compute figures are estimates.

## The idea

This is RL² ([Duan et al., 2016](https://arxiv.org/abs/1611.02779)) with evolution strategies (ES) as the training loop.

- **Slow loop (ES):** changes the weights of a recurrent network across generations, using only the reward.
- **Fast loop (inside one trial):** the weights are frozen. Any learning happens in the network's hidden state as it plays several puzzles in a row.
- No backprop anywhere.

## What the world has to do

Standard Sudoku rules never change, so the slow loop would bake them in and the fast loop would have nothing to learn. Each trial therefore has a **hidden extra rule**, found only through penalties. Simulating ideal agents before building anything showed three more requirements:

| Finding | What it changes |
|---|---|
| With a mixed list of named rules (diagonals, corners, anti-knight and so on), memory is worth only about 6 points to an ideal agent at 14 clues. It already scores about 91% with no memory | Use one family of rules instead: **no same digit at a hidden offset (dr, dc)**. All 18 offsets tested have valid 6x6 grids |
| With that family at 14 clues, an ideal agent that forgets between puzzles scores about 78%. One that remembers scores 79% on puzzle 1, 95% on puzzle 2 and 98-100% after that | Memory is worth about 20 points to an ideal agent, and most of it arrives by puzzle 3. Trials can be short |
| Many puzzles don't need the rule at all: about 12% at 12 clues, 37% at 14, 80% at 18 | Only keep puzzles where the hidden rule removes at least one solution that the base rules allow |

Without knowing the rule, an ideal solver scores about 46% on the kept puzzles at 14 clues.

**Open problem, to fix before any training.** The ideal agents above search every possible solution. An agent that only checks whether the next move is legal gets **no gain from memory**: at 14 clues it scores about 55% with memory, about 55% without, and 53% even when told the rule. Its failures are dead ends that need lookahead, which a small GRU is unlikely to do. Options to test: restrict the mask to digits that appear in some valid completion, or score conflicts instead of finished grids.

## Setup

| Part | Choice |
|---|---|
| Grid | 6x6 with 14 clues. 4x4 is for debugging only: it has 288 valid grids, few enough to memorize |
| Trial | 4 puzzles with the same hidden offset. Hidden state carries across puzzles and resets between trials |
| Cursor | The environment picks the blank with the fewest options under the base rules, so it can't reveal the hidden rule |
| Action | Choose a digit, masked to digits legal under row, column and box |
| Input per step | Grid contents, fixed cells, cursor, legal mask, previous action, previous reward, done flag |
| Reward | +0.1 per safe placement, -0.1 if it breaks the hidden rule (ends the puzzle), +1 for a finished grid |
| Network | GRU with about 64 units |

The mask and cursor are deliberate. A small flat GRU probably can't do Sudoku deduction on its own, and a failure there would say nothing about adaptation. They are not enough yet (see the open problem above).

## Training

Standard ES: mirrored noise, rank-transformed returns, weight decay. Start with a population of 256-1024 for 500-3000 generations.

From [Salimans et al., 2017](https://arxiv.org/abs/1703.03864):

- ES used 3x to 10x as much data as A3C on most Atari games, and up to 10x as much as TRPO on MuJoCo.
- The noise must actually change behaviour, or every copy scores the same and there is nothing to select.
- Its networks were feed-forward as far as I read, so ES on a GRU is untested by that paper. Stage 0 checks it.

## Controls

| Control | What it rules out |
|---|---|
| Same GRU, hidden state reset every puzzle | No learning across puzzles (the key control) |
| Reward and previous-action inputs zeroed | Adaptation without feedback |
| Feed-forward net trained the same way | Memory isn't needed |
| Random choice among legal digits | Floor |
| Network told the hidden offset | Ceiling for this network |

## Pass rules, fixed before running

12 offsets are for training and 6 are held out. Mirror-image offsets stay on the same side. On held-out offsets, over at least 1000 trials, all of these must hold:

1. Success rises from puzzle 1 to puzzles 3-4, with a confidence interval that excludes zero.
2. Success on puzzles 2-4 beats the reset control by at least 5 points.
3. Zeroing the reward input at test time removes the gain.
4. Swapping in hidden state from a trial with a different offset hurts performance.

If training offsets show adaptation and held-out offsets don't, that is a **fail**: the slow loop memorized them. Entirely new kinds of rule (diagonals, for example) are a stretch test, not part of the pass.

## Stages

Stop at the first gate that fails.

| Stage | Task | Where | Gate |
|---|---|---|---|
| W | Change the world, with scripted agents and no network | Laptop | A one-step agent gains at least 5 points from memory |
| 0 | 10-armed bandit, ES on the GRU | Laptop | Roughly matches UCB |
| 1 | 4x4, fixed rules, ES | Laptop | Solves above 90% |
| 2 | 6x6, 3 hidden offsets | Kaggle T4 | Beats the reset control |
| 3 | 6x6, all 12 training offsets | Kaggle T4 | Beats the reset control |
| 4 | 6x6, held-out offsets, all controls | Kaggle T4 | All four pass rules |

A failure at stage 0 means the problem is ES on a GRU, not Sudoku.

## Main risks

- ES stalls on the noisy return of a roughly 90-step trial.
- The network adapts on training offsets but not held-out ones.
- One puzzle gives away most of the rule, so the gain over the reset control is small.
- The legal mask makes the task easy enough that this says little about learning Sudoku itself. That is accepted: the question here is adaptation, not solving.

## Prior work

RL² and the main text of the ES paper were read. The rest were checked at abstract level or less, so treat these as pointers.

- **RL²** (1611.02779): GRU with 256 units trained with TRPO, on bandits, small MDPs and mazes. It trailed the baselines on the largest settings, and the authors blame the outer RL algorithm.
- **Evolution as the meta-learning loop:** 1812.11314, 2109.10781, and 2112.08588 (with plasticity).
- **In-context RL:** Algorithm Distillation (2210.14215), Decision-Pretrained Transformer (2306.14892).
- **Sudoku, learned with supervision:** Recurrent Relational Networks (1711.08028), SATNet (1905.12149), Hierarchical Reasoning Model (2506.21734).

A brief search found no work combining ES, RL² and Sudoku. That search was not thorough, so this repo makes no novelty claim.

## History

An earlier Sudoku playground is in git history at commit `a0e138f`. It had a feed-forward net fill the whole board at once, with a 0/1 reward. When every noisy copy scored 0, it moved toward the first copy, so at that point training was a random walk. It saved no results.
