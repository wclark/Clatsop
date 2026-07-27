# Clatsop

`clatsop` is a small Python package for representing American checkers
boards, turns, primitive rules, and rulesets using native 32-bit playable-square
masks.

The package is currently focused on a clean inspectable core:

- `Board`: black, white, and king 32-bit masks;
- `Turn`: a `Board`, side to move, and optional metadata;
- `Rule`: three condition masks plus four packed metadata bits;
- `Ruleset`: a collection of rules with native tuple, dict, and set views;
- notebook-friendly board and rule visualizations.

## Local Development

From this repository root:

```powershell
python -m pip install -e ".[dev]"
python -m compileall -q src tests benchmarks
ruff check src tests benchmarks
ruff format --check src tests benchmarks
pyright
pytest --cov=clatsop --cov-report=term-missing
mkdocs build --strict
python benchmarks/benchmark_move_generation.py --iterations 10000
```

The CI path uses `pytest` with coverage reporting and `ruff` for lint and
format checks. API docs are built with MkDocs and mkdocstrings.

## Minimal Example

```python
from clatsop import Ruleset, Turn

ruleset = Ruleset.american()
turn = Turn.initial()

rules = ruleset.to_dataframe()
transitions = ruleset.legal_transitions(turn)
successors = ruleset.successors(turn)
king_captures = ruleset.rules_for(king=True, capture=True)

print(len(rules), len(transitions), len(successors), len(king_captures))
print(ruleset.has_capture(turn), ruleset.has_promotion(turn))
turn.display()
ruleset.display(
    promotion=True,
    sort_by=("black_to_move", "capture"),
)
```

## Native Data

The core classes expose Python-native structures for analysis and tablebase
experiments:

```python
board_key = turn.board.key
turn_record = turn.as_dict()
rule_records = ruleset.records
rule_index = ruleset.record_map[0]
rule_keys = ruleset.rule_set
by_side = ruleset.rules_by_side
by_metadata = ruleset.rules_by_metadata
packed_state = turn.packed
```

All masks are 32-bit playable-square masks. There is no 64-bit board conversion
layer in the public API.

Inside a `Ruleset`, the three mask columns are contiguous unsigned 32-bit
arrays and flags are a contiguous unsigned-byte array. `Rule` objects,
dictionaries, and dataframes are inspection views created on demand.
`ruleset.buffers` returns immutable typed snapshots for NumPy, Numba, or
compiled consumers without exposing mutable engine storage.

Candidate rows are prefiltered by side, presence of a king belonging to the
moving side, promotion policy, and capture policy. Runtime matching and effects
then use integer bitwise operations. Each materialized `Rule` exposes
`is_black`, `is_white`, `is_empty`, and a packed `flags` integer. Applying a
rule derives its effects directly:

```python
cleared = rule.is_black | rule.is_white
destination = rule.is_empty
```

The `king` flag is a requirement, not a description of the moving piece.
`king=0` rules can be used by men or kings. `king=1` is stored only for
backward movement and for an existing king reaching the far rank without
promotion. A `promotion=1` rule instead requires a man and makes it a king.
This leaves 366 primitive American-checkers rules.

`legal_rule_indices()` and `legal_rules()` describe legal primitive first
steps. `legal_transitions()` and `successors()` expand mandatory capture chains
and only return completed turns. Promotion ends a capture chain.

For tablebase loops, a complete state is one packed 97-bit Python integer.
`expand_packed()` validates external input, avoids model-object allocation, and
reports subspace exits in the same pass:

```python
from clatsop import CAPTURE_AVAILABLE, PROMOTION_AVAILABLE, TERMINAL

next_states, status = ruleset.expand_packed(
    turn.packed,
    allow_captures=False,
    allow_promotions=False,
)

capture_exit = bool(status & CAPTURE_AVAILABLE)
promotion_exit = bool(status & PROMOTION_AVAILABLE)
terminal = bool(status & TERMINAL)
```

Generated states can use `expand_packed_unchecked()` inside a trusted hot loop
to avoid repeating input validation. `expand_packed_edges()` additionally
returns each complete primitive-rule path so predecessor relationships are not
lost. See the [tablebase guide](https://wclark.github.io/Clatsop/tablebases/).

`has_capture()` stops at the first matching capture and is suitable when only a
forced-capture probe is needed.

## Notebooks

- `notebooks/clatsop_examples.ipynb` is a game-level visual walkthrough of
  turns, legal rules, completed transitions, filtered rule displays, and the
  full ruleset.
- `notebooks/clatsop_rule_catalog.ipynb` is a minimal standalone visual catalog
  of all primitive rules.

Notebook sources are committed without generated images. CI executes both
notebooks and uploads the complete rendered notebooks as build artifacts.
