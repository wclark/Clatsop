"""Native 32-bit playable-square encodings for checkers boards."""

from operator import index as integer_index
from typing import SupportsIndex, cast

MASK32 = (1 << 32) - 1


def is_playable_square(row: object, col: object) -> bool:
    """Return whether ``(row, col)`` is one of the 32 playable board squares."""

    try:
        row = integer_index(cast(SupportsIndex, row))
        col = integer_index(cast(SupportsIndex, col))
    except TypeError:
        return False
    return 0 <= row < 8 and 0 <= col < 8 and (row + col) % 2 == 1


def square_index32(row: object, col: object) -> int:
    """Return the 0-based playable-square index for ``(row, col)``."""

    try:
        row = integer_index(cast(SupportsIndex, row))
        col = integer_index(cast(SupportsIndex, col))
    except TypeError as error:
        raise ValueError(f"square coordinates must be integers: {(row, col)}") from error
    if not is_playable_square(row, col):
        raise ValueError(f"square is not playable: {(row, col)}")
    return row * 4 + (col // 2)


def square_mask32(row: object, col: object) -> int:
    """Return a one-bit 32-bit mask for a playable board square."""

    return 1 << square_index32(row, col)


def square_coords32(square: object) -> tuple[int, int]:
    """Return ``(row, col)`` for a playable-square index."""

    try:
        square = integer_index(cast(SupportsIndex, square))
    except TypeError as error:
        raise ValueError(f"square must be an integer: {square!r}") from error
    if not (0 <= square < 32):
        raise ValueError(f"square out of range: {square!r}")
    row, offset = divmod(square, 4)
    col = 2 * offset + (1 if row % 2 == 0 else 0)
    return row, col


def square_from_mask32(mask: object) -> tuple[int, int]:
    """Return ``(row, col)`` for a one-bit 32-bit playable-square mask."""

    try:
        mask = integer_index(cast(SupportsIndex, mask))
    except TypeError as error:
        raise ValueError(f"mask must be an integer: {mask!r}") from error
    if mask <= 0 or mask & (mask - 1):
        raise ValueError("mask must contain exactly one bit")
    square = mask.bit_length() - 1
    if square >= 32:
        raise ValueError("mask is outside the playable-square board")
    return square_coords32(square)


def playable_squares() -> tuple[tuple[int, int, int], ...]:
    """Return all playable squares as ``(row, col, mask)`` tuples."""

    return tuple(
        (row, col, square_mask32(row, col)) for row in range(8) for col in range(8) if is_playable_square(row, col)
    )
