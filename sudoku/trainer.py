from __future__ import annotations

import math
import random
import threading
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence

from sudoku_core import SudokuCurriculum, flatten, is_valid_solution, puzzle_mask, unflatten


def softmax(logits: Sequence[float], temperature: float) -> List[float]:
    adjusted = [value / max(temperature, 1e-6) for value in logits]
    offset = max(adjusted)
    exp_values = [math.exp(value - offset) for value in adjusted]
    total = sum(exp_values) or 1.0
    return [value / total for value in exp_values]


def sample_index(probabilities: Sequence[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(probabilities) - 1


def activation(name: str, value: float) -> float:
    if name == "tanh":
        return math.tanh(value)
    if name == "relu":
        return value if value > 0 else 0.0
    if name == "sin":
        return math.sin(value)
    return value / (1.0 + abs(value))


@dataclass
class Architecture:
    hidden_sizes: List[int]
    activation: str
    learning_rate: float
    noise_scale: float
    temperature: float
    batch_size: int
    steps: int
    clue_schedule: List[int]
    init_scale: float
    elite_bias: float


def random_architecture(rng: random.Random) -> Architecture:
    depth = rng.choice([1, 2, 2, 3])
    hidden_sizes = [rng.choice([48, 64, 72, 96, 128]) for _ in range(depth)]
    schedule_length = rng.choice([3, 4, 5])
    start = rng.randint(70, 76)
    schedule = [max(26, start - 4 * index) for index in range(schedule_length)]
    return Architecture(
        hidden_sizes=hidden_sizes,
        activation=rng.choice(["tanh", "relu", "softsign", "sin"]),
        learning_rate=rng.uniform(0.02, 0.16),
        noise_scale=rng.uniform(0.04, 0.22),
        temperature=rng.uniform(0.45, 1.25),
        batch_size=rng.choice([8, 12, 16, 20]),
        steps=rng.choice([24, 32, 40, 48, 56]),
        clue_schedule=schedule,
        init_scale=rng.uniform(0.03, 0.22),
        elite_bias=rng.uniform(0.15, 0.5),
    )


def mutate_architecture(parent: Architecture, rng: random.Random) -> Architecture:
    child = deepcopy(parent)
    mutation_count = rng.randint(2, 5)
    for _ in range(mutation_count):
        choice = rng.choice(
            ["hidden", "activation", "lr", "noise", "temp", "batch", "steps", "schedule", "scale", "elite"]
        )
        if choice == "hidden":
            if rng.random() < 0.35 and len(child.hidden_sizes) < 4:
                child.hidden_sizes.append(rng.choice([48, 64, 96, 128]))
            elif rng.random() < 0.25 and len(child.hidden_sizes) > 1:
                child.hidden_sizes.pop(rng.randrange(len(child.hidden_sizes)))
            else:
                index = rng.randrange(len(child.hidden_sizes))
                child.hidden_sizes[index] = rng.choice([48, 64, 72, 96, 128, 144])
        elif choice == "activation":
            child.activation = rng.choice(["tanh", "relu", "softsign", "sin"])
        elif choice == "lr":
            child.learning_rate = min(0.25, max(0.005, child.learning_rate * rng.uniform(0.75, 1.35)))
        elif choice == "noise":
            child.noise_scale = min(0.35, max(0.01, child.noise_scale * rng.uniform(0.7, 1.4)))
        elif choice == "temp":
            child.temperature = min(1.8, max(0.25, child.temperature * rng.uniform(0.75, 1.25)))
        elif choice == "batch":
            child.batch_size = rng.choice([6, 8, 10, 12, 16, 20, 24])
        elif choice == "steps":
            child.steps = rng.choice([20, 24, 32, 40, 48, 56, 64])
        elif choice == "schedule":
            pivot = rng.randrange(len(child.clue_schedule))
            child.clue_schedule[pivot] = min(80, max(24, child.clue_schedule[pivot] + rng.choice([-4, -2, 2, 4])))
            child.clue_schedule = sorted(child.clue_schedule, reverse=True)
        elif choice == "scale":
            child.init_scale = min(0.3, max(0.01, child.init_scale * rng.uniform(0.75, 1.3)))
        elif choice == "elite":
            child.elite_bias = min(0.8, max(0.05, child.elite_bias * rng.uniform(0.8, 1.2)))
    return child


class Layer2Learner:
    def __init__(self, architecture: Architecture, seed: int) -> None:
        self.arch = architecture
        self.rng = random.Random(seed)
        self.shapes = [162] + architecture.hidden_sizes + [81 * 9]
        self.weights: List[List[List[float]]] = []
        self.biases: List[List[float]] = []
        for input_size, output_size in zip(self.shapes[:-1], self.shapes[1:]):
            self.weights.append(
                [
                    [self.rng.uniform(-architecture.init_scale, architecture.init_scale) for _ in range(input_size)]
                    for _ in range(output_size)
                ]
            )
            self.biases.append(
                [self.rng.uniform(-architecture.init_scale, architecture.init_scale) for _ in range(output_size)]
            )

    def clone_parameters(self) -> Dict[str, List]:
        return {"weights": deepcopy(self.weights), "biases": deepcopy(self.biases)}

    def load_parameters(self, params: Dict[str, List]) -> None:
        self.weights = deepcopy(params["weights"])
        self.biases = deepcopy(params["biases"])

    def _forward_logits(self, puzzle_vector: Sequence[float], mask_vector: Sequence[float]) -> List[float]:
        layer = list(puzzle_vector) + list(mask_vector)
        for layer_index, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            next_layer = []
            is_last = layer_index == len(self.weights) - 1
            for neuron_weights, bias in zip(weights, biases):
                total = bias
                for weight, value in zip(neuron_weights, layer):
                    total += weight * value
                next_layer.append(total if is_last else activation(self.arch.activation, total))
            layer = next_layer
        return layer

    def solve(self, puzzle_grid: Sequence[Sequence[int]]) -> Dict[str, object]:
        puzzle = flatten(puzzle_grid)
        mask = puzzle_mask(puzzle_grid)
        normalized_puzzle = [value / 9.0 for value in puzzle]
        logits = self._forward_logits(normalized_puzzle, mask)

        output_digits: List[int] = []
        confidence = 0.0
        for cell_index in range(81):
            if puzzle[cell_index]:
                output_digits.append(puzzle[cell_index])
                confidence += 1.0
                continue
            start = cell_index * 9
            probs = softmax(logits[start:start + 9], self.arch.temperature)
            pick = sample_index(probs, self.rng)
            output_digits.append(pick + 1)
            confidence += max(probs)

        return {"board": unflatten(output_digits), "avg_confidence": confidence / 81.0}

    def jitter_from(self, params: Dict[str, List]) -> Dict[str, List]:
        noisy = {"weights": deepcopy(params["weights"]), "biases": deepcopy(params["biases"])}
        for layer in noisy["weights"]:
            for neuron in layer:
                for weight_index, value in enumerate(neuron):
                    neuron[weight_index] = value + self.rng.gauss(0.0, self.arch.noise_scale)
        for layer in noisy["biases"]:
            for bias_index, value in enumerate(layer):
                layer[bias_index] = value + self.rng.gauss(0.0, self.arch.noise_scale)
        return noisy

    def update_towards(self, base: Dict[str, List], elite: Dict[str, List]) -> Dict[str, List]:
        updated = {"weights": deepcopy(base["weights"]), "biases": deepcopy(base["biases"])}
        rate = self.arch.learning_rate
        bias = self.arch.elite_bias

        for base_layer, elite_layer in zip(updated["weights"], elite["weights"]):
            for base_neuron, elite_neuron in zip(base_layer, elite_layer):
                for weight_index, value in enumerate(base_neuron):
                    direction = elite_neuron[weight_index] - value
                    base_neuron[weight_index] = value + direction * rate * (1.0 + bias)

        for base_layer, elite_layer in zip(updated["biases"], elite["biases"]):
            for bias_index, value in enumerate(base_layer):
                direction = elite_layer[bias_index] - value
                base_layer[bias_index] = value + direction * rate * (1.0 + bias)

        return updated


@dataclass
class LifeMetrics:
    generation: int
    architecture: Dict[str, object]
    lifetime_steps: int
    total_successes: int
    success_rate: float
    best_reward: float
    best_clue_count: int
    timestamp: float
    trace: List[Dict[str, object]] = field(default_factory=list)
    best_board: List[List[int]] = field(default_factory=list)
    best_puzzle: List[List[int]] = field(default_factory=list)


class MetaSudokuTrainer:
    def __init__(self) -> None:
        self.curriculum = SudokuCurriculum(seed=17)
        self.meta_rng = random.Random(123)
        self.lock = threading.Lock()
        self.state = {
            "running": False,
            "generation": 0,
            "best_generation": None,
            "lives": [],
            "current": None,
            "started_at": None,
        }
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        with self.lock:
            self.state["running"] = True
            self.state["started_at"] = time.time()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            return deepcopy(self.state)

    def _loop(self) -> None:
        champion = random_architecture(self.meta_rng)
        champion_score = -1.0
        generation = 0

        while True:
            with self.lock:
                if not self.state["running"]:
                    return

            generation += 1
            architecture = champion if generation == 1 else mutate_architecture(champion, self.meta_rng)
            metrics = self._run_life(generation, architecture)
            score = metrics["success_rate"] * 100.0 + metrics["best_reward"] * 10.0

            if score >= champion_score:
                champion = architecture
                champion_score = score
                with self.lock:
                    self.state["best_generation"] = generation

            with self.lock:
                self.state["generation"] = generation
                self.state["current"] = metrics
                self.state["lives"].append(metrics)
                self.state["lives"] = self.state["lives"][-18:]

    def _run_life(self, generation: int, architecture: Architecture) -> Dict[str, object]:
        learner = Layer2Learner(architecture, seed=1000 + generation)
        params = learner.clone_parameters()
        trace: List[Dict[str, object]] = []
        successes = 0
        best_reward = 0.0
        best_board: List[List[int]] = []
        best_puzzle: List[List[int]] = []
        best_clue_count = architecture.clue_schedule[0]

        for step in range(architecture.steps):
            schedule_index = min(step * len(architecture.clue_schedule) // architecture.steps, len(architecture.clue_schedule) - 1)
            clue_count = architecture.clue_schedule[schedule_index]
            batch_samples = [self.curriculum.sample(clue_count) for _ in range(architecture.batch_size)]
            best_candidate = None
            best_candidate_reward = -1.0
            best_confidence = 0.0

            for sample in batch_samples:
                candidate_params = learner.jitter_from(params)
                learner.load_parameters(candidate_params)
                rollout = learner.solve(sample.puzzle)
                solved = is_valid_solution(rollout["board"], sample.puzzle)
                reward = 1.0 if solved else 0.0

                if reward > best_candidate_reward:
                    best_candidate = {
                        "params": candidate_params,
                        "sample": sample,
                        "rollout": rollout,
                        "reward": reward,
                    }
                    best_candidate_reward = reward
                    best_confidence = rollout["avg_confidence"]

                if reward > best_reward:
                    best_reward = reward
                    best_board = rollout["board"]
                    best_puzzle = sample.puzzle
                    best_clue_count = sample.clue_count

            assert best_candidate is not None
            params = learner.update_towards(params, best_candidate["params"])
            learner.load_parameters(params)

            if best_candidate["reward"] > 0:
                successes += 1

            trace.append(
                {
                    "step": step + 1,
                    "clues": clue_count,
                    "reward": best_candidate["reward"],
                    "successes": successes,
                    "avg_confidence": round(best_confidence, 4),
                    "board": best_candidate["rollout"]["board"],
                    "puzzle": best_candidate["sample"].puzzle,
                }
            )

        metrics = LifeMetrics(
            generation=generation,
            architecture=asdict(architecture),
            lifetime_steps=architecture.steps,
            total_successes=successes,
            success_rate=successes / max(architecture.steps, 1),
            best_reward=best_reward,
            best_clue_count=best_clue_count,
            timestamp=time.time(),
            trace=trace[-24:],
            best_board=best_board,
            best_puzzle=best_puzzle,
        )
        return asdict(metrics)