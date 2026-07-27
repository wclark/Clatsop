"""Validated and packed native checkers state keys."""

from __future__ import annotations

from operator import index as integer_index
from typing import SupportsIndex, TypeAlias, cast

from .encoding import MASK32

StateKey: TypeAlias = tuple[int, int, int, int]
PackedState: TypeAlias = int

PACKED_STATE_BITS = 97
PACKED_STATE_MASK = (1 << PACKED_STATE_BITS) - 1
STANDARD_PIECES_PER_SIDE = 12
INITIAL_QUIET_PROGRESS = 24


def mask32(value: object, name: str) -> int:
    """Return an integer-like value after enforcing the 32-bit mask range."""

    value = _integer(value, name)
    if value < 0 or value & ~MASK32:
        raise ValueError(f"{name} must fit in a 32-bit playable-square bitmap")
    return value


def turn_bit(value: object) -> int:
    """Normalize a side name or integer-like bit."""

    if value in ("B", "black", "Black"):
        return 1
    if value in ("W", "white", "White"):
        return 0
    try:
        side = integer_index(cast(SupportsIndex, value))
    except TypeError as error:
        raise ValueError("side to move must be 0, 1, 'black', or 'white'") from error
    if side not in (0, 1):
        raise ValueError("side to move must be 0, 1, 'black', or 'white'")
    return side


def validate_board_masks(black: object, white: object, kings: object = 0) -> tuple[int, int, int]:
    """Validate and return the three board masks."""

    black = mask32(black, "black")
    white = mask32(white, "white")
    kings = mask32(kings, "kings")
    if black & white:
        raise ValueError("black and white masks overlap")
    if kings & ~(black | white):
        raise ValueError("kings must be a subset of occupied squares")
    return black, white, kings


def validate_state_values(
    black: object,
    white: object,
    kings: object,
    black_to_move: object,
) -> StateKey:
    """Validate and return a native four-integer state key."""

    black, white, kings = validate_board_masks(black, white, kings)
    return black, white, kings, turn_bit(black_to_move)


def pack_state_values(
    black: object,
    white: object,
    kings: object,
    black_to_move: object,
) -> PackedState:
    """Validate and pack a state into one 97-bit Python integer."""

    black, white, kings, black_to_move = validate_state_values(black, white, kings, black_to_move)
    return pack_validated_state(black, white, kings, black_to_move)


def pack_validated_state(black: int, white: int, kings: int, black_to_move: int) -> PackedState:
    """Pack already-validated state values without allocating a model object."""

    return black | (white << 32) | (kings << 64) | (black_to_move << 96)


def unpack_state_values(state: object) -> StateKey:
    """Validate and unpack one 97-bit state integer."""

    state = _integer(state, "packed state")
    if state < 0 or state & ~PACKED_STATE_MASK:
        raise ValueError("packed state must fit in 97 bits")
    black, white, kings, black_to_move = unpack_validated_state(state)
    if black & white:
        raise ValueError("black and white masks overlap")
    if kings & ~(black | white):
        raise ValueError("kings must be a subset of occupied squares")
    return black, white, kings, black_to_move


def unpack_validated_state(state: int) -> StateKey:
    """Unpack an already-validated 97-bit state integer."""

    return (
        state & MASK32,
        (state >> 32) & MASK32,
        (state >> 64) & MASK32,
        (state >> 96) & 1,
    )


def quiet_ply_values(black: int, white: int, kings: int, black_to_move: int) -> int:
    """Return the unique quiet-move layer relative to the standard initial state."""

    black, white, kings, black_to_move = validate_state_values(black, white, kings, black_to_move)
    if black.bit_count() != STANDARD_PIECES_PER_SIDE or white.bit_count() != STANDARD_PIECES_PER_SIDE:
        raise ValueError("quiet ply requires twelve pieces per side")
    if kings:
        raise ValueError("quiet ply is defined only for positions without kings")

    progress = 0
    for row in range(8):
        row_mask = 0xF << (4 * row)
        progress += row * (black & row_mask).bit_count()
        progress += (7 - row) * (white & row_mask).bit_count()

    ply = progress - INITIAL_QUIET_PROGRESS
    if black_to_move != int(ply % 2 == 0):
        raise ValueError("side to move is inconsistent with the quiet ply")
    return ply


def _integer(value: object, name: str) -> int:
    try:
        return integer_index(cast(SupportsIndex, value))
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
