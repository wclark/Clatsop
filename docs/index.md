# Clatsop

`clatsop` is a compact native-32-bit model for American checkers boards,
turns, primitive rules, and rulesets.

The core public objects are:

- `Board`, a tuple-like set of black, white, and king masks.
- `Turn`, a `Board` plus side-to-move and optional metadata.
- `Rule`, three condition masks plus packed move metadata.
- `Ruleset`, the 366-rule American checkers primitive-rule table with
  filterable and sortable visual displays.
- Packed 97-bit state keys and allocation-light expansion methods for
  tablebase generation.

The validated model API is intended for notebooks and inspection. A separate
packed-state path supports large hash-based searches while preserving the same
rules and mandatory-capture behavior.
