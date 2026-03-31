from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence

Grid = List[List[int]]


BASE_SOLUTION: Grid = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]


def clone_grid(grid: Sequence[Sequence[int]]) -> Grid:
    return [list(row) for row in grid]


def flatten(grid: Sequence[Sequence[int]]) -> List[int]:
    return [value for row in grid for value in row]


def unflatten(values: Sequence[int]) -> Grid:
    return [list(values[index:index + 9]) for index in range(0, 81, 9)]


def _permute_digits(grid: Grid, rng: random.Random) -> Grid:
    mapping = list(range(10))
    digits = list(range(1, 10))
    shuffled = digits[:]
    rng.shuffle(shuffled)
    for source, target in zip(digits, shuffled):
        mapping[source] = target
    return [[mapping[value] for value in row] for row in grid]


def _shuffle_rows_and_columns(grid: Grid, rng: random.Random) -> Grid:
    board = clone_grid(grid)

    row_bands = [board[index:index + 3] for index in range(0, 9, 3)]
    rng.shuffle(row_bands)
    shuffled_rows: Grid = []
    for band in row_bands:
        band_rows = band[:]
        rng.shuffle(band_rows)
        shuffled_rows.extend(band_rows)

    columns = list(zip(*shuffled_rows))
    col_stacks = [list(columns[index:index + 3]) for index in range(0, 9, 3)]
    rng.shuffle(col_stacks)
    shuffled_cols = []
    for stack in col_stacks:
        stack_cols = stack[:]
        rng.shuffle(stack_cols)
        shuffled_cols.extend(stack_cols)

    return [list(row) for row in zip(*shuffled_cols)]


def generate_solution(rng: random.Random) -> Grid:
    grid = _permute_digits(clone_grid(BASE_SOLUTION), rng)
    return _shuffle_rows_and_columns(grid, rng)


def make_puzzle(solution: Grid, clue_count: int, rng: random.Random) -> Grid:
    values = flatten(solution)
    indexes = list(range(81))
    rng.shuffle(indexes)
    keep = set(indexes[:clue_count])
    return unflatten([value if index in keep else 0 for index, value in enumerate(values)])


def puzzle_mask(puzzle: Grid) -> List[int]:
    return [1 if value else 0 for value in flatten(puzzle)]


def board_satisfies_puzzle(board: Grid, puzzle: Grid) -> bool:
    for row_index in range(9):
        for col_index in range(9):
            clue = puzzle[row_index][col_index]
            if clue and board[row_index][col_index] != clue:
                return False
    return True


def _all_groups(grid: Grid) -> List[List[int]]:
    groups: List[List[int]] = []
    groups.extend([row[:] for row in grid])
    groups.extend([list(column) for column in zip(*grid)])
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            box = []
            for row_index in range(box_row, box_row + 3):
                for col_index in range(box_col, box_col + 3):
                    box.append(grid[row_index][col_index])
            groups.append(box)
    return groups


def is_valid_solution(board: Grid, puzzle: Grid) -> bool:
    if not board_satisfies_puzzle(board, puzzle):
        return False
    expected = list(range(1, 10))
    for group in _all_groups(board):
        if sorted(group) != expected:
            return False
    return True


@dataclass
class PuzzleSample:
    puzzle: Grid
    solution: Grid
    clue_count: int


class SudokuCurriculum:
    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)

    def sample(self, clue_count: int) -> PuzzleSample:
        solution = generate_solution(self.rng)
        puzzle = make_puzzle(solution, clue_count=clue_count, rng=self.rng)
        return PuzzleSample(puzzle=puzzle, solution=solution, clue_count=clue_count)