"""Clatsop: native 32-bit American checkers rules and board states."""

from importlib.metadata import PackageNotFoundError, version

from .display import draw_board, show_board, show_ruleset_rows, show_turn
from .encoding import (
    MASK32,
    is_playable_square,
    playable_squares,
    square_coords32,
    square_from_mask32,
    square_index32,
    square_mask32,
)
from .ruleset import (
    CAPTURE_AVAILABLE,
    PROMOTION_AVAILABLE,
    TERMINAL,
    Board,
    Rule,
    Ruleset,
    Transition,
    Turn,
    american_ruleset,
    as_board,
    as_turn,
    pack_state,
    quiet_ply,
    unpack_state,
)
from .state import PACKED_STATE_BITS, PACKED_STATE_MASK, PackedState, StateKey

try:
    __version__ = version("clatsop")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "Board",
    "CAPTURE_AVAILABLE",
    "MASK32",
    "PACKED_STATE_BITS",
    "PACKED_STATE_MASK",
    "PackedState",
    "PROMOTION_AVAILABLE",
    "Rule",
    "Ruleset",
    "StateKey",
    "Transition",
    "Turn",
    "TERMINAL",
    "american_ruleset",
    "as_board",
    "as_turn",
    "draw_board",
    "is_playable_square",
    "pack_state",
    "playable_squares",
    "quiet_ply",
    "show_board",
    "show_ruleset_rows",
    "show_turn",
    "square_coords32",
    "square_from_mask32",
    "square_index32",
    "square_mask32",
    "unpack_state",
]
