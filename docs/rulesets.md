# Rulesets

The default ruleset is `Ruleset.american()`. It stores one row per primitive
runtime rule. Each `Rule` stores exactly three 32-bit masks and one four-bit
flags integer:

- masks: `is_black`, `is_white`, `is_empty`;
- flags: `black_to_move`, `king`, `promotion`, `capture`.

The `king` flag means that the moving piece must be a king. An unset flag adds
no king restriction, so the same forward rule applies to either a man or a
king. King-specific rules are generated only for backward movement and the
far-rank case where an existing king must not promote. Conversely,
`promotion=1` requires a man. The complete American ruleset therefore contains
366 primitive rules.

## Runtime storage

`Ruleset` stores masks in three contiguous unsigned 32-bit arrays and packed
flags in an unsigned-byte array. It does not retain hundreds of `Rule` model
objects. Those objects, records, dataframes, and displays are generated only
when requested.

The packed columns can be consumed as immutable typed snapshots:

```python
buffers = ruleset.buffers

is_black = buffers["is_black"]
is_white = buffers["is_white"]
is_empty = buffers["is_empty"]
flags = buffers["flags"]
```

The returned memory views have immutable `bytes` owners, so they cannot mutate
the ruleset or desynchronize its runtime indexes. Candidate rows are
prefiltered by side, whether the moving side has any kings, whether promotion
rules are allowed, and capture policy. The remaining condition checks and
effects use integer bit operations.

The dataframe expands the packed flags into separate columns for analysis.
Effects are not stored as additional masks:

```python
cleared = rule.is_black | rule.is_white
destination = rule.is_empty
```

The ruleset exposes native Python views for analysis:

```python
from clatsop import Ruleset

ruleset = Ruleset.american()

records = ruleset.records
record_map = ruleset.record_map
rule_keys = ruleset.rule_set
by_side = ruleset.rules_by_side
by_metadata = ruleset.rules_by_metadata
```

It also provides dataframe and graphical display views. Metadata filters and
sorting are available directly on the ruleset:

```python
df = ruleset.to_dataframe()
figures = ruleset.display(df.iloc[:3])

king_captures = ruleset.rules_for(
    black_to_move="white",
    king=True,
    capture=True,
    sort_by=("promotion", "is_empty"),
    reverse=True,
)
figures = ruleset.display(
    promotion=True,
    sort_by=("black_to_move", "capture"),
)
```

The supported filters are `black_to_move`, `king`, `promotion`, and `capture`.
`sort_by` accepts one field name or a tuple of names, including `rule_index`
and any dataframe column. `select_records(...)` and `indices_for(...)` provide
the same filtering and ordering for dictionary records and stable rule indexes.

Construct compact custom rules with `Rule.from_masks(...)`:

```python
from clatsop import Rule, square_mask32

rule = Rule.from_masks(
    is_black=square_mask32(2, 1),
    is_white=0,
    is_empty=square_mask32(3, 0),
    black_to_move=1,
)
```

Legal primitive rules are matched against `Turn` objects:

```python
from clatsop import Ruleset, Turn

ruleset = Ruleset.american()
turn = Turn.initial()

rule_indexes = ruleset.legal_rule_indices(turn)
rules = ruleset.legal_rules(turn)
transitions = ruleset.legal_transitions(turn)
next_turns = ruleset.successors(turn)
next_by_rule = ruleset.successor_map(turn)
```

Rule methods return legal primitive first steps. Transition and successor
methods finish every mandatory capture chain before changing sides. Their
mapping keys are tuples of stable primitive rule indexes, one index per jump.

The separate [tablebase guide](tablebases.md) covers packed states, status
flags, predecessor edges, and quiet-layer generation.
