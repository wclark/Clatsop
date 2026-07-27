from collections import Counter

from hypothesis import given, settings
from hypothesis import strategies as st

from clatsop import (
    CAPTURE_AVAILABLE,
    MASK32,
    PROMOTION_AVAILABLE,
    TERMINAL,
    Ruleset,
    Turn,
    pack_state,
    square_index32,
)

RULESET = Ruleset.american()
DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
INDEX_BY_COORD = {(row, col): square_index32(row, col) for row in range(8) for col in range(8) if (row + col) % 2 == 1}
COORD_BY_INDEX = {index: coord for coord, index in INDEX_BY_COORD.items()}


@st.composite
def valid_state_keys(draw):
    black = draw(st.integers(min_value=0, max_value=MASK32))
    white = draw(st.integers(min_value=0, max_value=MASK32)) & ~black
    kings = draw(st.integers(min_value=0, max_value=MASK32)) & (black | white)
    side = draw(st.integers(min_value=0, max_value=1))
    return black, white, kings, side


@given(valid_state_keys())
@settings(max_examples=1_000, deadline=None)
def test_engine_matches_independent_reference(state):
    expected, expected_status = _reference_expansion(state)
    expected_counts = Counter(expected)

    turn = Turn.from_tuple(state)
    transitions = RULESET.legal_transitions(turn)
    expanded, status = RULESET.expand_keys(state)
    packed, packed_status = RULESET.expand_packed(pack_state(state))
    edges, edge_status = RULESET.expand_edges(state)

    assert Counter(transition.turn.key for transition in transitions) == expected_counts
    assert Counter(expanded) == expected_counts
    assert Counter(state for _path, state in edges) == expected_counts
    assert Counter(packed) == Counter(pack_state(state) for state in expected)
    assert status == expected_status
    assert packed_status == expected_status
    assert edge_status == expected_status


def _reference_expansion(state):
    black, white, kings, black_to_move = state
    captures = _capture_steps(black, white, kings, black_to_move)
    if captures:
        successors = []
        promotion_available = False
        for step in captures:
            nested, promoted = _finish_capture_step(step, black_to_move)
            successors.extend(nested)
            promotion_available |= promoted
        status = CAPTURE_AVAILABLE
        if promotion_available:
            status |= PROMOTION_AVAILABLE
        return successors, status

    quiet = _quiet_steps(black, white, kings, black_to_move)
    if not quiet:
        return [], TERMINAL
    status = PROMOTION_AVAILABLE if any(promoted for _state, promoted in quiet) else 0
    return [state for state, _promoted in quiet], status


def _finish_capture_step(step, black_to_move):
    black, white, kings, destination, promoted = step
    if promoted:
        return [(black, white, kings, 1 - black_to_move)], True

    captures = _capture_steps(
        black,
        white,
        kings,
        black_to_move,
        only_piece=destination,
    )
    if not captures:
        return [(black, white, kings, 1 - black_to_move)], False

    successors = []
    promotion_available = False
    for nested_step in captures:
        nested, nested_promoted = _finish_capture_step(nested_step, black_to_move)
        successors.extend(nested)
        promotion_available |= nested_promoted
    return successors, promotion_available


def _capture_steps(black, white, kings, black_to_move, only_piece=None):
    own = black if black_to_move else white
    opponent = white if black_to_move else black
    occupied = black | white
    steps = []

    for origin in _bits(own):
        if only_piece is not None and origin != only_piece:
            continue
        origin_index = origin.bit_length() - 1
        row, col = COORD_BY_INDEX[origin_index]
        is_king = bool(kings & origin)
        for dr, dc in _directions(black_to_move, is_king):
            middle_index = INDEX_BY_COORD.get((row + dr, col + dc))
            destination_index = INDEX_BY_COORD.get((row + 2 * dr, col + 2 * dc))
            if middle_index is None or destination_index is None:
                continue
            middle = 1 << middle_index
            destination = 1 << destination_index
            if not opponent & middle or occupied & destination:
                continue
            steps.append(
                _apply_reference_step(
                    black,
                    white,
                    kings,
                    black_to_move,
                    origin,
                    destination,
                    middle,
                    is_king,
                )
            )
    return steps


def _quiet_steps(black, white, kings, black_to_move):
    own = black if black_to_move else white
    occupied = black | white
    steps = []

    for origin in _bits(own):
        origin_index = origin.bit_length() - 1
        row, col = COORD_BY_INDEX[origin_index]
        is_king = bool(kings & origin)
        for dr, dc in _directions(black_to_move, is_king):
            destination_index = INDEX_BY_COORD.get((row + dr, col + dc))
            if destination_index is None:
                continue
            destination = 1 << destination_index
            if occupied & destination:
                continue
            next_black, next_white, next_kings, _destination, promoted = _apply_reference_step(
                black,
                white,
                kings,
                black_to_move,
                origin,
                destination,
                0,
                is_king,
            )
            steps.append(
                (
                    (next_black, next_white, next_kings, 1 - black_to_move),
                    promoted,
                )
            )
    return steps


def _apply_reference_step(
    black,
    white,
    kings,
    black_to_move,
    origin,
    destination,
    captured,
    is_king,
):
    cleared = origin | captured
    if black_to_move:
        next_black = (black & ~cleared) | destination
        next_white = white & ~cleared
    else:
        next_black = black & ~cleared
        next_white = (white & ~cleared) | destination

    destination_row, _ = COORD_BY_INDEX[destination.bit_length() - 1]
    promoted = not is_king and destination_row == (7 if black_to_move else 0)
    next_kings = kings & ~cleared
    if is_king or promoted:
        next_kings |= destination
    return next_black, next_white, next_kings, destination, promoted


def _directions(black_to_move, is_king):
    forward = 1 if black_to_move else -1
    if is_king:
        return DIRECTIONS
    return ((forward, -1), (forward, 1))


def _bits(mask):
    while mask:
        bit = mask & -mask
        yield bit
        mask ^= bit
