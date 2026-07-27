"""Small repeatable baseline for Clatsop's pure-Python move generator."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from sys import getsizeof
from time import perf_counter

from clatsop import CAPTURE_AVAILABLE, PROMOTION_AVAILABLE, Ruleset, Turn, square_mask32


def benchmark(iterations: int) -> dict[str, float | int]:
    ruleset = Ruleset.american()
    initial = Turn.initial()
    turns = (
        initial,
        Turn.from_masks(square_mask32(2, 1), square_mask32(3, 2), 0, 1),
        Turn.from_masks(
            square_mask32(2, 1),
            square_mask32(3, 2) | square_mask32(5, 4),
            0,
            1,
        ),
        Turn.from_masks(square_mask32(6, 1), 0, 0, 1),
        Turn.from_masks(square_mask32(2, 1), 0, square_mask32(2, 1), 1),
        Turn.from_masks(initial.black, initial.white, initial.white, 1),
        Turn.from_masks(initial.black, initial.white, initial.black, 1),
        Turn.from_masks(square_mask32(7, 0), 0, 0, 1),
    )

    transition_started = perf_counter()
    transitions = 0
    for _ in range(iterations):
        for turn in turns:
            transitions += len(ruleset.legal_transitions(turn))
    transition_seconds = perf_counter() - transition_started

    positions = iterations * len(turns)
    key_metrics = _measure_expansion(
        tuple(turn.key for turn in turns),
        ruleset.expand_keys,
        iterations,
    )
    packed_metrics = _measure_expansion(
        tuple(turn.packed for turn in turns),
        ruleset.expand_packed,
        iterations,
    )
    trusted_metrics = _measure_expansion(
        tuple(turn.packed for turn in turns),
        ruleset.expand_packed_unchecked,
        iterations,
    )
    tuple_key = turns[0].key
    packed_key = turns[0].packed
    return {
        "iterations": iterations,
        "cases": len(turns),
        "positions": positions,
        "transitions": transitions,
        "transition_seconds": transition_seconds,
        "transition_positions_per_second": positions / transition_seconds,
        "validated_key_seconds": key_metrics["seconds"],
        "validated_key_positions_per_second": positions / key_metrics["seconds"],
        "validated_packed_seconds": packed_metrics["seconds"],
        "validated_packed_positions_per_second": positions / packed_metrics["seconds"],
        "trusted_packed_seconds": trusted_metrics["seconds"],
        "trusted_packed_positions_per_second": positions / trusted_metrics["seconds"],
        "tablebase_successors": trusted_metrics["successors"],
        "tablebase_exits": trusted_metrics["exits"],
        "tuple_key_payload_bytes": getsizeof(tuple_key) + sum(getsizeof(value) for value in tuple_key),
        "packed_key_payload_bytes": getsizeof(packed_key),
    }


def _measure_expansion(
    states: tuple[object, ...],
    expand: Callable[..., tuple[tuple[object, ...], int]],
    iterations: int,
) -> dict[str, float | int]:
    started = perf_counter()
    successors = 0
    exits = 0
    for _ in range(iterations):
        for state in states:
            next_states, status = expand(
                state,
                allow_captures=False,
                allow_promotions=False,
            )
            successors += len(next_states)
            exits += bool(status & (CAPTURE_AVAILABLE | PROMOTION_AVAILABLE))
    return {
        "seconds": perf_counter() - started,
        "successors": successors,
        "exits": exits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.iterations), indent=2))


if __name__ == "__main__":
    main()
