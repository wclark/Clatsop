import importlib
import sys
import types
import unittest
from importlib.metadata import PackageNotFoundError
from unittest.mock import Mock, patch

import matplotlib

import clatsop

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from clatsop import (
    CAPTURE_AVAILABLE,
    MASK32,
    PROMOTION_AVAILABLE,
    TERMINAL,
    Board,
    Rule,
    Ruleset,
    Transition,
    Turn,
    as_board,
    as_turn,
    is_playable_square,
    pack_state,
    playable_squares,
    quiet_ply,
    show_board,
    show_ruleset_rows,
    show_turn,
    square_coords32,
    square_from_mask32,
    square_index32,
    square_mask32,
    unpack_state,
)
from clatsop.display import _display_figure, _show_rule_row


class NativeApiTests(unittest.TestCase):
    def test_source_tree_version_fallback(self):
        with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            reloaded = importlib.reload(clatsop)
            self.assertEqual(reloaded.__version__, "0+unknown")
        self.assertNotEqual(importlib.reload(clatsop).__version__, "0+unknown")

    def test_square_encoding_is_native_32_bit(self):
        self.assertEqual(len(playable_squares()), 32)
        self.assertTrue(is_playable_square(0, 1))
        self.assertFalse(is_playable_square(0, 0))
        self.assertEqual(square_index32(0, 1), 0)
        self.assertEqual(square_index32(7, 6), 31)
        self.assertEqual(square_coords32(0), (0, 1))
        self.assertEqual(square_coords32(31), (7, 6))
        self.assertEqual(square_from_mask32(square_mask32(3, 4)), (3, 4))
        with self.assertRaises(ValueError):
            square_index32(8, 1)
        with self.assertRaises(ValueError):
            square_coords32(32)
        with self.assertRaises(ValueError):
            square_from_mask32(0)
        with self.assertRaises(ValueError):
            square_from_mask32(3)
        with self.assertRaises(ValueError):
            square_from_mask32(1 << 32)
        with self.assertRaises(ValueError):
            square_index32(0.5, 1)
        with self.assertRaises(ValueError):
            square_coords32(1.5)
        with self.assertRaises(ValueError):
            square_from_mask32(1.5)
        self.assertFalse(is_playable_square(0.5, 1))

    def test_board_and_turn_are_hashable_native_records(self):
        board = Board.initial()
        turn = Turn.initial()
        metadata_turn = Turn(board, "white", {"round": 3, "kind": "demo"})

        self.assertEqual(board.as_tuple(), (0x00000FFF, 0xFFF00000, 0))
        self.assertEqual(board.key, board.as_tuple())
        self.assertEqual(board.maps, {"black": 0x00000FFF, "white": 0xFFF00000, "kings": 0})
        self.assertEqual(board.occupied, 0xFFF00FFF)
        self.assertEqual(board.empty, MASK32 ^ 0xFFF00FFF)
        self.assertEqual(turn.as_tuple(), (0x00000FFF, 0xFFF00000, 0, 1))
        self.assertEqual(turn.key, turn.as_tuple())
        self.assertEqual(turn.maps, turn.as_dict())
        self.assertEqual(turn.occupied, board.occupied)
        self.assertEqual(turn.empty, board.empty)
        self.assertEqual(turn.side, "black")
        self.assertEqual(metadata_turn.side, "white")
        self.assertEqual(dict(metadata_turn.metadata), {"kind": "demo", "round": 3})
        self.assertEqual(metadata_turn.as_tuple(), (*board.as_tuple(), 0))
        self.assertEqual(metadata_turn.key, Turn(board, "white").key)
        self.assertEqual(metadata_turn, Turn(board, "white"))
        self.assertEqual(len({metadata_turn, Turn(board, "white", {"path": [1, 2]})}), 1)
        self.assertEqual(metadata_turn.as_dict()["metadata"], {"kind": "demo", "round": 3})
        self.assertEqual(as_board(board.as_dict()), board)
        self.assertEqual(as_turn(turn.as_dict()), turn)
        self.assertEqual(as_turn(turn.as_tuple()), turn)
        self.assertEqual(Turn.from_tuple((*board.as_tuple(), 0)).side, "white")
        self.assertEqual(Turn(board, 1, [("source", "list")]).metadata, (("source", "list"),))

        with self.assertRaises(ValueError):
            Board(1, 1, 0)
        with self.assertRaises(ValueError):
            Board(1, 0, 2)
        with self.assertRaises(ValueError):
            Board(1 << 32, 0, 0)
        with self.assertRaises(ValueError):
            Board(1.5, 0, 0)
        with self.assertRaises(ValueError):
            Board.from_tuple((1, 2))
        with self.assertRaises(ValueError):
            Turn.from_tuple((1, 2, 3))
        with self.assertRaises(ValueError):
            Turn.from_tuple((*board.as_tuple(), 1, {"x": 1}))
        with self.assertRaises(ValueError):
            Turn(board, 1, ("not-a-pair",))
        with self.assertRaises(ValueError):
            Turn(Board(0, 0, 0), "red")

    def test_packed_states_validate_and_round_trip(self):
        turn = Turn.initial()
        packed = pack_state(turn)

        self.assertEqual(turn.packed, packed)
        self.assertEqual(unpack_state(packed), turn.key)
        self.assertEqual(Turn.from_packed(packed), turn)
        self.assertEqual(Turn.from_packed(packed, {"source": "packed"}).maps["metadata"], {"source": "packed"})

        invalid_keys = (
            (1, 1, 0, 1),
            (1, 0, 2, 1),
            (1 << 32, 0, 0, 1),
            (1, 0, 0, 2),
            (1.5, 0, 0, 1),
        )
        ruleset = Ruleset.american()
        self.assertEqual(
            ruleset.expand_packed_unchecked(packed),
            ruleset.expand_packed(packed),
        )
        for key in invalid_keys:
            with self.subTest(key=key), self.assertRaises(ValueError):
                pack_state(key)
            with self.subTest(expand=key), self.assertRaises(ValueError):
                ruleset.expand_keys(key)

        with self.assertRaises(ValueError):
            unpack_state(1 << 97)
        with self.assertRaises(ValueError):
            unpack_state(1 | (1 << 32))
        with self.assertRaises(ValueError):
            unpack_state(2 << 64)
        with self.assertRaises(ValueError):
            unpack_state(1.5)

    def test_quiet_ply_proves_layered_acyclic_search(self):
        ruleset = Ruleset.american()
        frontier = {Turn.initial().packed}
        seen = set()

        for expected_ply in range(8):
            self.assertTrue(frontier)
            self.assertTrue(frontier.isdisjoint(seen))
            self.assertTrue(all(quiet_ply(unpack_state(state)) == expected_ply for state in frontier))
            seen.update(frontier)
            next_frontier = set()
            for state in frontier:
                successors, _status = ruleset.expand_packed(
                    state,
                    allow_captures=False,
                    allow_promotions=False,
                )
                next_frontier.update(successors)
            frontier = next_frontier

        with self.assertRaises(ValueError):
            quiet_ply(Turn(Board.initial(), 0))
        with self.assertRaises(ValueError):
            quiet_ply(Turn.from_masks(1, 0, 0, 1))
        with self.assertRaises(ValueError):
            quiet_ply(Turn.from_masks(Board.initial().black, Board.initial().white, 1, 1))

    def test_ruleset_geometry_and_native_indexes(self):
        ruleset = Ruleset.american()
        dataframe = ruleset.to_dataframe()

        self.assertIs(ruleset, Ruleset.american())
        self.assertEqual(len(ruleset), 366)
        self.assertEqual(dataframe.shape, (366, 7))
        self.assertEqual(len(ruleset.records), 366)
        self.assertEqual(len(ruleset.record_map), 366)
        self.assertEqual(len(ruleset.rule_set), 366)
        self.assertEqual(len(ruleset.rules_by_side[1]), 183)
        self.assertEqual(len(ruleset.rules_by_side[0]), 183)
        self.assertEqual(dataframe["black_to_move"].sum(), 183)
        self.assertEqual(dataframe["king"].sum(), 196)
        self.assertEqual(dataframe["promotion"].sum(), 26)
        self.assertEqual(dataframe["capture"].sum(), 156)
        self.assertEqual(len(ruleset.indices_for(black_to_move=1, king=0, promotion=0, capture=0)), 42)
        self.assertEqual(len(ruleset.indices_for(capture=1)), 156)
        self.assertEqual(len(ruleset.indices_for(king=1, promotion=1)), 0)
        self.assertEqual(len(Ruleset.american("black")), 183)
        self.assertEqual(len(Ruleset.american("white")), 183)
        self.assertEqual(len(Ruleset.from_board_geometry(side=None)), 366)
        self.assertEqual(next(iter(ruleset)), ruleset[0])
        self.assertIsNot(next(iter(ruleset)), ruleset[0])

        buffers = ruleset.buffers
        self.assertEqual(set(buffers), {"is_black", "is_white", "is_empty", "flags"})
        self.assertTrue(all(buffer.readonly for buffer in buffers.values()))
        self.assertEqual(buffers["is_black"].itemsize, 4)
        self.assertEqual(buffers["flags"].itemsize, 1)
        self.assertEqual(buffers["is_black"][0], square_mask32(0, 1))
        self.assertEqual(buffers["flags"][0], 1)
        self.assertTrue(all(isinstance(buffer.obj, bytes) for buffer in buffers.values()))
        with self.assertRaises(TypeError):
            buffers["is_black"].obj[0] = 0

        records = ruleset.records
        record_map = ruleset.record_map
        rule_set = ruleset.rule_set
        side_map = ruleset.rules_by_side
        metadata_map = ruleset.rules_by_metadata
        records[0]["is_black"] = 0
        record_map[0]["is_black"] = 0
        rule_set.clear()
        side_map.clear()
        metadata_map.clear()
        self.assertNotEqual(ruleset.record(0)["is_black"], 0)
        self.assertEqual(len(ruleset.rule_set), 366)
        self.assertEqual(len(ruleset.rules_by_side), 2)
        self.assertTrue(ruleset.rules_by_metadata)

        first = ruleset.record(0)
        self.assertEqual(first["is_black"], square_mask32(0, 1))
        self.assertEqual(first["is_white"], 0)
        self.assertEqual(first["is_empty"], square_mask32(1, 0))
        self.assertEqual(first["black_to_move"], 1)
        self.assertEqual(first["king"], 0)
        self.assertEqual(first["promotion"], 0)
        self.assertEqual(first["capture"], 0)
        self.assertEqual(len(ruleset[0].as_tuple()), 4)

        with self.assertRaises(IndexError):
            ruleset.record(-1)
        with self.assertRaises(IndexError):
            _ = ruleset[9999]
        with self.assertRaises(ValueError):
            _ = ruleset[1.5]
        with self.assertRaises(ValueError):
            Ruleset.american("red")
        with self.assertRaisesRegex(ValueError, "duplicate rule"):
            Ruleset.from_records([ruleset[0], ruleset[0]])

    def test_every_generated_rule_obeys_independent_mask_invariants(self):
        ruleset = Ruleset.american()

        for index, rule in enumerate(ruleset):
            with self.subTest(rule=index):
                condition_masks = (rule.is_black, rule.is_white, rule.is_empty)

                self.assertFalse(condition_masks[0] & condition_masks[1])
                self.assertFalse(condition_masks[0] & condition_masks[2])
                self.assertFalse(condition_masks[1] & condition_masks[2])
                self.assertEqual(rule.mover.bit_count(), 1)
                self.assertEqual(rule.is_empty.bit_count(), 1)
                from_row, from_col = square_from_mask32(rule.mover)
                to_row, to_col = square_from_mask32(rule.is_empty)
                self.assertEqual(abs(to_row - from_row), 2 if rule.capture else 1)
                self.assertEqual(abs(to_col - from_col), 2 if rule.capture else 1)
                opponent = rule.is_white if rule.black_to_move else rule.is_black
                self.assertEqual(opponent.bit_count(), rule.capture)
                self.assertEqual(bool(rule.capture), bool(opponent))
                promotion_rank = 0xF0000000 if rule.black_to_move else 0x0000000F
                self.assertEqual(
                    bool(rule.promotion),
                    not rule.king and bool(rule.is_empty & promotion_rank),
                )
                cleared = rule.is_black | rule.is_white
                self.assertTrue(cleared & rule.mover)
                self.assertFalse(cleared & rule.is_empty)

                kings = rule.mover if rule.king else 0
                minimal_turn = Turn.from_masks(
                    rule.is_black,
                    rule.is_white,
                    kings,
                    rule.black_to_move,
                )
                self.assertTrue(rule.applies(minimal_turn))
                result = rule.apply(minimal_turn)
                self.assertEqual(result.board.occupied.bit_count(), 1)
                self.assertTrue(result.board.occupied & rule.is_empty)
                self.assertEqual(bool(result.kings), bool(rule.king or rule.promotion))

                opposite_king_state = Turn.from_masks(
                    rule.is_black,
                    rule.is_white,
                    rule.mover,
                    rule.black_to_move,
                )
                if rule.king:
                    self.assertFalse(rule.applies(Turn.from_masks(rule.is_black, rule.is_white, 0, rule.black_to_move)))
                elif rule.promotion:
                    self.assertFalse(rule.applies(opposite_king_state))
                else:
                    self.assertTrue(rule.applies(opposite_king_state))
                    self.assertEqual(rule.apply(opposite_king_state).kings, rule.is_empty)

    def test_only_necessary_king_specific_geometries_are_duplicated(self):
        ruleset = Ruleset.american()
        by_geometry = {}

        for rule in ruleset:
            key = (rule.is_black, rule.is_white, rule.is_empty, rule.black_to_move)
            by_geometry.setdefault(key, []).append(rule)

            from_row, _ = square_from_mask32(rule.mover)
            to_row, _ = square_from_mask32(rule.is_empty)
            dr = to_row - from_row
            moves_forward = dr > 0 if rule.black_to_move else dr < 0
            destination_rank = 0xF0000000 if rule.black_to_move else 0x0000000F
            if rule.king:
                self.assertTrue(not moves_forward or rule.is_empty & destination_rank)
            else:
                self.assertTrue(moves_forward)

        duplicates = [rules for rules in by_geometry.values() if len(rules) > 1]
        self.assertEqual(len(duplicates), 26)
        for rules in duplicates:
            self.assertEqual(len(rules), 2)
            self.assertEqual(
                {(rule.king, rule.promotion) for rule in rules},
                {(0, 1), (1, 0)},
            )

    def test_rule_masks_flags_and_records(self):
        rule = Rule.from_masks(
            square_mask32(2, 1),
            0,
            square_mask32(3, 0),
            black_to_move=1,
        )
        turn = Turn.from_masks(square_mask32(2, 1), 0, 0, 1)
        next_turn = rule.apply(turn)

        self.assertTrue(rule.applies(turn))
        king_turn = Turn.from_masks(square_mask32(2, 1), 0, square_mask32(2, 1), 1)
        self.assertTrue(rule.applies(king_turn))
        self.assertEqual(rule.apply(king_turn).kings, square_mask32(3, 0))
        self.assertEqual(rule.as_tuple()[0], square_mask32(2, 1))
        self.assertEqual(rule.flags, 1)
        self.assertEqual(rule.black_to_move, 1)
        self.assertEqual(rule.king, 0)
        self.assertEqual(rule.promotion, 0)
        self.assertEqual(rule.capture, 0)
        self.assertEqual(next_turn.black, square_mask32(3, 0))
        self.assertEqual(next_turn.black_to_move, 0)
        self.assertEqual(rule.apply(turn, switch_side=False).black_to_move, 1)

        copied = Rule.from_record(rule.as_dict())
        packed = Rule.from_record(
            {
                "is_black": rule.is_black,
                "is_white": rule.is_white,
                "is_empty": rule.is_empty,
                "flags": rule.flags,
            }
        )
        white_rule = Rule.from_masks(
            0,
            square_mask32(5, 0),
            square_mask32(4, 1),
            black_to_move="W",
        )
        self.assertEqual(copied.as_tuple(), rule.as_tuple())
        self.assertEqual(packed.as_tuple(), rule.as_tuple())
        self.assertEqual(white_rule.mover, square_mask32(5, 0))
        self.assertEqual(white_rule.flags, 0)
        self.assertFalse(rule.applies(Turn.from_masks(square_mask32(2, 1), 0, 0, 0)))
        self.assertFalse(
            rule.applies(
                Turn.from_masks(
                    square_mask32(2, 1) | square_mask32(3, 0),
                    0,
                    0,
                    1,
                )
            )
        )
        capture_rule = Rule.from_masks(
            square_mask32(2, 1),
            square_mask32(3, 2),
            square_mask32(4, 3),
            black_to_move=1,
        )
        self.assertFalse(capture_rule.applies(Turn.from_masks(square_mask32(2, 1), 0, 0, 1)))
        with self.assertRaises(ValueError):
            rule.apply(Turn.from_masks(0, 0, 0, 1))

    def test_capture_king_rule_metadata(self):
        rule = Rule.from_masks(
            square_mask32(4, 1),
            square_mask32(5, 0),
            square_mask32(3, 2),
            black_to_move=0,
            king=1,
        )
        turn = Turn.from_masks(square_mask32(4, 1), square_mask32(5, 0), square_mask32(5, 0), 0)
        next_turn = rule.apply(turn)

        self.assertTrue(rule.applies(turn))
        self.assertEqual(rule.flags, 0b1010)
        self.assertEqual(rule.is_black, square_mask32(4, 1))
        self.assertEqual(next_turn.white, square_mask32(3, 2))
        self.assertEqual(next_turn.kings, square_mask32(3, 2))

    def test_rule_validation_rejects_inconsistent_compact_records(self):
        origin = square_mask32(0, 1)
        destination = square_mask32(1, 0)

        for flags in (-1, 1 << 4):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                Rule(origin, 0, destination, flags)

        overlapping_masks = (
            (origin, origin, destination, 1),
            (origin, 0, origin, 1),
            (0, origin, origin, 0),
        )
        for values in overlapping_masks:
            with self.subTest(overlap=values), self.assertRaises(ValueError):
                Rule(*values)

        with self.assertRaises(ValueError):
            Rule(origin, 0, destination, 1.5)
        with self.assertRaises(ValueError):
            Rule.from_masks(origin, 0, destination, 1, king=2)
        with self.assertRaises(ValueError):
            Rule.from_masks(origin, 0, destination, 1, capture="yes")

        for mover in (0, origin | square_mask32(0, 3)):
            with self.subTest(mover=mover), self.assertRaises(ValueError):
                Rule(mover, 0, destination, 1)

        for empty in (0, destination | square_mask32(1, 2)):
            with self.subTest(empty=empty), self.assertRaises(ValueError):
                Rule(origin, 0, empty, 1)

        with self.assertRaises(ValueError):
            Rule(origin, 0, destination, 0b1001)
        with self.assertRaises(ValueError):
            Rule(origin, square_mask32(1, 2), square_mask32(2, 3), 0b0001)

        with self.assertRaises(ValueError):
            Rule(square_mask32(6, 1), 0, square_mask32(7, 0), 0b0001)
        with self.assertRaises(ValueError):
            Rule(origin, 0, destination, 0b0101)

    def test_initial_legal_rules_and_successors(self):
        ruleset = Ruleset.american()
        turn = Turn.initial()
        rules = ruleset.legal_rules(turn)
        successors = ruleset.successors(turn)
        successor_map = ruleset.successor_map(turn)

        self.assertEqual(len(rules), 7)
        self.assertEqual(len(successors), 7)
        self.assertEqual(len(successor_map), 7)
        self.assertTrue(all(not rule.capture for rule in rules))
        self.assertEqual(ruleset.legal_rule_indices(turn), ruleset.matching_rule_indices(turn))
        self.assertEqual(len(ruleset.matching_rules(turn)), 7)
        self.assertEqual(len(ruleset.legal_rules(turn, from_mask=square_mask32(2, 1))), 2)

    def test_forced_capture_suppresses_quiet_moves(self):
        ruleset = Ruleset.american()
        turn = Turn.from_masks(square_mask32(2, 1), square_mask32(3, 2), 0, 1)
        rules = ruleset.legal_rules(turn)
        next_turn = ruleset.successors(turn)[0]

        self.assertEqual(len(rules), 1)
        self.assertTrue(rules[0].capture)
        self.assertTrue(ruleset.matching_rule_indices(turn, capture=True))
        self.assertEqual(rules[0].is_black, square_mask32(2, 1))
        self.assertEqual(rules[0].is_empty, square_mask32(4, 3))
        self.assertEqual(rules[0].is_white, square_mask32(3, 2))
        self.assertEqual(next_turn.black, square_mask32(4, 3))
        self.assertEqual(next_turn.white, 0)
        self.assertEqual(next_turn.black_to_move, 0)

    def test_forced_capture_is_global_when_filtering_by_piece(self):
        ruleset = Ruleset.american()
        capturing_piece = square_mask32(2, 1)
        quiet_piece = square_mask32(2, 5)
        turn = Turn.from_masks(capturing_piece | quiet_piece, square_mask32(3, 2), 0, 1)

        self.assertTrue(ruleset.legal_rule_indices(turn, from_mask=capturing_piece))
        self.assertEqual(ruleset.legal_rule_indices(turn, from_mask=quiet_piece), [])
        self.assertEqual(ruleset.legal_rules(turn, from_mask=quiet_piece), ())

    def test_capture_chains_are_completed_before_switching_sides(self):
        ruleset = Ruleset.american()
        turn = Turn.from_masks(
            square_mask32(2, 1),
            square_mask32(3, 2) | square_mask32(5, 4),
            0,
            1,
        )

        transitions = ruleset.legal_transitions(turn)
        self.assertEqual(len(transitions), 1)
        self.assertIsInstance(transitions[0], Transition)
        self.assertEqual(len(transitions[0].rule_indices), 2)
        self.assertEqual(transitions[0].turn.black, square_mask32(6, 5))
        self.assertEqual(transitions[0].turn.white, 0)
        self.assertEqual(transitions[0].turn.black_to_move, 0)
        self.assertEqual(transitions[0].key, (transitions[0].rule_indices, transitions[0].turn.key))
        self.assertEqual(transitions[0].as_dict()["turn"], transitions[0].turn.as_dict())
        self.assertEqual(ruleset.successors(turn), (transitions[0].turn,))
        self.assertEqual(ruleset.successor_map(turn), {transitions[0].rule_indices: transitions[0].turn})

    def test_promotion_ends_a_capture_chain(self):
        ruleset = Ruleset.american()
        remaining_white = square_mask32(6, 3)
        turn = Turn.from_masks(
            square_mask32(5, 0),
            square_mask32(6, 1) | remaining_white,
            0,
            1,
        )

        transitions = ruleset.legal_transitions(turn)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(len(transitions[0].rule_indices), 1)
        self.assertEqual(transitions[0].turn.black, square_mask32(7, 2))
        self.assertEqual(transitions[0].turn.white, remaining_white)
        self.assertEqual(transitions[0].turn.kings, square_mask32(7, 2))
        self.assertEqual(transitions[0].turn.black_to_move, 0)

        chained_turn = Turn.from_masks(
            square_mask32(3, 0),
            square_mask32(4, 1) | square_mask32(6, 3),
            0,
            1,
        )
        chained_transition = ruleset.legal_transitions(chained_turn)[0]
        self.assertEqual(len(chained_transition.rule_indices), 2)
        self.assertEqual(chained_transition.turn.black, square_mask32(7, 4))
        self.assertEqual(chained_transition.turn.kings, square_mask32(7, 4))

    def test_terminal_turn_has_no_transitions(self):
        ruleset = Ruleset.american()
        turn = Turn.from_masks(square_mask32(7, 0), 0, 0, 1)

        self.assertEqual(ruleset.legal_transitions(turn), ())
        self.assertEqual(ruleset.successors(turn), ())
        self.assertEqual(ruleset.successor_map(turn), {})
        with self.assertRaises(ValueError):
            Transition((), turn)
        with self.assertRaises(ValueError):
            Transition((1.5,), turn)

    def test_king_and_promotion_semantics(self):
        ruleset = Ruleset.american()
        king_turn = Turn.from_masks(square_mask32(2, 1), 0, square_mask32(2, 1), 1)
        promotion_turn = Turn.from_masks(square_mask32(6, 1), 0, 0, 1)

        king_rules = ruleset.legal_rules(king_turn)
        promotion_indexes = ruleset.legal_rule_indices(promotion_turn)
        promotion_successors = ruleset.successors(promotion_turn)

        self.assertEqual(len(king_rules), 4)
        self.assertTrue(
            any(square_from_mask32(rule.is_empty)[0] < square_from_mask32(rule.mover)[0] for rule in king_rules)
        )
        self.assertEqual(len(promotion_indexes), 2)
        self.assertTrue(all(ruleset[index].promotion for index in promotion_indexes))
        self.assertTrue(all(successor.kings & successor.black for successor in promotion_successors))
        self.assertTrue(any(ruleset[index].flags & 0b1111 for index in promotion_indexes))

    def test_runtime_policy_filters_and_fast_probes(self):
        ruleset = Ruleset.american()
        initial = Turn.initial()
        capture_from = square_mask32(2, 1)
        capture_turn = Turn.from_masks(capture_from, square_mask32(3, 2), 0, 1)
        promotion_turn = Turn.from_masks(square_mask32(6, 1), 0, 0, 1)
        man_turn = Turn.from_masks(square_mask32(2, 1), 0, 0, 1)
        king_turn = Turn.from_masks(square_mask32(2, 1), 0, square_mask32(2, 1), 1)

        self.assertFalse(ruleset.has_capture(initial))
        self.assertTrue(ruleset.has_capture(capture_turn))
        self.assertTrue(ruleset.has_capture(capture_turn.as_dict()))
        self.assertTrue(ruleset.has_capture(capture_turn.key, from_mask=capture_from))
        self.assertFalse(ruleset.has_capture(capture_turn, from_mask=square_mask32(2, 5)))
        self.assertFalse(ruleset.has_promotion(initial))
        self.assertTrue(ruleset.has_promotion(promotion_turn))

        captures = ruleset.matching_rule_indices(capture_turn, capture=True)
        quiet = ruleset.matching_rule_indices(capture_turn, capture=False)
        self.assertEqual(len(captures), 1)
        self.assertTrue(quiet)
        self.assertTrue(all(ruleset[index].capture for index in captures))
        self.assertTrue(all(not ruleset[index].capture for index in quiet))

        promotion_matches = ruleset.matching_rule_indices(promotion_turn)
        self.assertEqual(len(promotion_matches), 2)
        self.assertEqual(
            ruleset.matching_rule_indices(promotion_turn, allow_promotions=False),
            [],
        )
        self.assertEqual(len(ruleset.matching_rule_indices(man_turn)), 2)
        self.assertEqual(len(ruleset.matching_rule_indices(king_turn)), 4)

        man_with_other_king = Turn.from_masks(
            square_mask32(4, 3) | square_mask32(0, 1),
            square_mask32(3, 2),
            square_mask32(0, 1),
            1,
        )
        self.assertFalse(ruleset.has_capture(man_with_other_king))
        self.assertFalse(ruleset.matching_rule_indices(man_with_other_king, capture=True))

        king_capture_to_far_rank = Turn.from_masks(
            square_mask32(5, 0),
            square_mask32(6, 1),
            square_mask32(5, 0),
            1,
        )
        self.assertTrue(ruleset.has_capture(king_capture_to_far_rank))
        self.assertEqual(
            len(ruleset.matching_rule_indices(king_capture_to_far_rank, capture=True)),
            1,
        )

    def test_native_key_expansion_reports_subspace_exits(self):
        ruleset = Ruleset.american()
        initial = Turn.initial()
        capture_turn = Turn.from_masks(square_mask32(2, 1), square_mask32(3, 2), 0, 1)
        promotion_turn = Turn.from_masks(square_mask32(6, 1), 0, 0, 1)
        terminal_turn = Turn.from_masks(square_mask32(7, 0), 0, 0, 1)

        initial_keys, initial_status = ruleset.expand_keys(
            initial,
            allow_captures=False,
            allow_promotions=False,
        )
        capture_keys, capture_status = ruleset.expand_keys(
            capture_turn.key,
            allow_captures=False,
            allow_promotions=False,
        )
        promotion_keys, promotion_status = ruleset.expand_keys(
            promotion_turn,
            allow_captures=False,
            allow_promotions=False,
        )
        terminal_keys, terminal_status = ruleset.expand_keys(terminal_turn)

        self.assertEqual(len(initial_keys), 7)
        self.assertEqual(initial_status, 0)
        self.assertEqual(capture_keys, ())
        self.assertEqual(capture_status, CAPTURE_AVAILABLE)
        self.assertEqual(promotion_keys, ())
        self.assertEqual(promotion_status, PROMOTION_AVAILABLE)
        self.assertEqual(terminal_keys, ())
        self.assertEqual(terminal_status, TERMINAL)

        expected_capture_keys = tuple(turn.key for turn in ruleset.successors(capture_turn))
        generated_capture_keys, generated_capture_status = ruleset.expand_keys(capture_turn)
        self.assertEqual(generated_capture_keys, expected_capture_keys)
        self.assertEqual(generated_capture_status, CAPTURE_AVAILABLE)

        promoted_keys, promoted_status = ruleset.expand_keys(promotion_turn)
        self.assertEqual(len(promoted_keys), 2)
        self.assertEqual(promoted_status, PROMOTION_AVAILABLE)
        self.assertTrue(all(key[2] for key in promoted_keys))

        capture_promotion_turn = Turn.from_masks(
            square_mask32(5, 0),
            square_mask32(6, 1),
            0,
            1,
        )
        capture_promotion_keys, capture_promotion_status = ruleset.expand_keys(
            capture_promotion_turn,
            allow_captures=False,
            allow_promotions=False,
        )
        self.assertEqual(capture_promotion_keys, ())
        self.assertEqual(
            capture_promotion_status,
            CAPTURE_AVAILABLE | PROMOTION_AVAILABLE,
        )

        mixed_turn = Turn.from_masks(
            square_mask32(6, 1) | square_mask32(2, 5),
            0,
            0,
            1,
        )
        mixed_keys, mixed_status = ruleset.expand_keys(
            mixed_turn,
            allow_captures=False,
            allow_promotions=False,
        )
        self.assertEqual(mixed_status, PROMOTION_AVAILABLE)
        self.assertTrue(mixed_keys)
        self.assertTrue(all(not key[2] for key in mixed_keys))

    def test_native_key_expansion_handles_capture_chain_promotions(self):
        ruleset = Ruleset.american()
        turn = Turn.from_masks(
            square_mask32(3, 0),
            square_mask32(4, 1) | square_mask32(6, 3),
            0,
            1,
        )

        pruned_keys, pruned_status = ruleset.expand_keys(
            turn,
            allow_captures=True,
            allow_promotions=False,
        )
        expanded_keys, expanded_status = ruleset.expand_keys(turn)

        self.assertEqual(pruned_keys, ())
        self.assertEqual(pruned_status, CAPTURE_AVAILABLE | PROMOTION_AVAILABLE)
        self.assertEqual(expanded_keys, tuple(successor.key for successor in ruleset.successors(turn)))
        self.assertEqual(expanded_status, CAPTURE_AVAILABLE | PROMOTION_AVAILABLE)

    def test_native_edge_expansion_preserves_complete_rule_paths(self):
        ruleset = Ruleset.american()
        initial = Turn.initial()
        initial_edges, initial_status = ruleset.expand_edges(initial)
        expected = tuple(
            (transition.rule_indices, transition.turn.key) for transition in ruleset.legal_transitions(initial)
        )

        self.assertEqual(initial_edges, expected)
        self.assertEqual(initial_status, 0)

        capture_turn = Turn.from_masks(
            square_mask32(2, 1),
            square_mask32(3, 2) | square_mask32(5, 4),
            0,
            1,
        )
        capture_edges, capture_status = ruleset.expand_edges(capture_turn)
        expected_capture = tuple(
            (transition.rule_indices, transition.turn.key) for transition in ruleset.legal_transitions(capture_turn)
        )
        self.assertEqual(capture_edges, expected_capture)
        self.assertEqual(capture_status, CAPTURE_AVAILABLE)
        self.assertTrue(all(len(path) == 2 for path, _state in capture_edges))

        packed_edges, packed_status = ruleset.expand_packed_edges(capture_turn.packed)
        self.assertEqual(
            packed_edges,
            tuple((path, pack_state(state)) for path, state in capture_edges),
        )
        self.assertEqual(packed_status, capture_status)

        pruned_edges, pruned_status = ruleset.expand_edges(capture_turn, allow_captures=False)
        self.assertEqual(pruned_edges, ())
        self.assertEqual(pruned_status, CAPTURE_AVAILABLE)

        promotion_turn = Turn.from_masks(square_mask32(6, 1), 0, 0, 1)
        promotion_edges, promotion_status = ruleset.expand_edges(promotion_turn)
        self.assertEqual(len(promotion_edges), 2)
        self.assertEqual(promotion_status, PROMOTION_AVAILABLE)
        self.assertTrue(all(state[2] for _path, state in promotion_edges))
        pruned_promotions, pruned_promotion_status = ruleset.expand_edges(
            promotion_turn,
            allow_promotions=False,
        )
        self.assertEqual(pruned_promotions, ())
        self.assertEqual(pruned_promotion_status, PROMOTION_AVAILABLE)

        capture_promotion_turn = Turn.from_masks(
            square_mask32(5, 0),
            square_mask32(6, 1),
            0,
            1,
        )
        capture_promotion_edges, capture_promotion_status = ruleset.expand_edges(capture_promotion_turn)
        self.assertEqual(len(capture_promotion_edges), 1)
        self.assertEqual(
            capture_promotion_status,
            CAPTURE_AVAILABLE | PROMOTION_AVAILABLE,
        )
        self.assertTrue(capture_promotion_edges[0][1][2])
        pruned_capture_promotions, pruned_capture_promotion_status = ruleset.expand_edges(
            capture_promotion_turn,
            allow_promotions=False,
        )
        self.assertEqual(pruned_capture_promotions, ())
        self.assertEqual(
            pruned_capture_promotion_status,
            CAPTURE_AVAILABLE | PROMOTION_AVAILABLE,
        )

        terminal_edges, terminal_status = ruleset.expand_edges(Turn.from_masks(square_mask32(7, 0), 0, 0, 1))
        self.assertEqual(terminal_edges, ())
        self.assertEqual(terminal_status, TERMINAL)

    def test_fast_rule_application_matches_object_api(self):
        ruleset = Ruleset.american()
        turns = (
            Turn.initial(),
            Turn(Board.initial(), 0),
            Turn.from_masks(square_mask32(2, 1), square_mask32(3, 2), 0, 1),
            Turn.from_masks(square_mask32(2, 1), 0, square_mask32(2, 1), 1),
        )

        for turn in turns:
            for index in ruleset.legal_rule_indices(turn):
                with self.subTest(turn=turn.key, rule=index):
                    expected = ruleset[index].apply(turn)
                    self.assertEqual(ruleset.apply_rule(turn, index), expected)
                    self.assertEqual(ruleset.apply_key(turn.key, index), expected.key)
                    self.assertEqual(ruleset.apply_packed(turn.packed, index), expected.packed)
                    self.assertEqual(
                        ruleset.apply_key(turn, index, switch_side=False)[3],
                        turn.black_to_move,
                    )

        with self.assertRaises(ValueError):
            ruleset.apply_rule(Turn.from_masks(0, 0, 0, 1), 0)
        with self.assertRaises(ValueError):
            ruleset.apply_key(Turn(Board.initial(), 0), 0)
        with self.assertRaises(ValueError):
            ruleset.apply_packed(Turn(Board.initial(), 0).packed, 0)

        first_rule = ruleset[0]
        occupied_destination = Turn.from_masks(
            first_rule.is_black | first_rule.is_empty,
            0,
            0,
            first_rule.black_to_move,
        )
        with self.assertRaises(ValueError):
            ruleset.apply_key(occupied_destination, 0)
        with self.assertRaises(ValueError):
            ruleset.apply_packed(1 | (1 << 32), 0)

    def test_ruleset_record_inputs_and_selection(self):
        ruleset = Ruleset.american()
        dataframe_ruleset = Ruleset.from_records(ruleset.to_dataframe().iloc[:2])
        rule_ruleset = Ruleset.from_records(ruleset[0])
        dict_ruleset = Ruleset.from_records(ruleset.record(0))
        list_ruleset = Ruleset.from_records([ruleset.record(0), ruleset.record(1)])
        copied_ruleset = Ruleset.from_records(dataframe_ruleset)

        self.assertEqual(len(dataframe_ruleset), 2)
        self.assertEqual(len(rule_ruleset), 1)
        self.assertEqual(len(dict_ruleset), 1)
        self.assertEqual(len(list_ruleset), 2)
        self.assertEqual(len(copied_ruleset), 2)
        self.assertEqual(len(Ruleset.from_records(ruleset.to_dataframe().iloc[0])), 1)
        self.assertEqual(len(ruleset.select_records(None)), 366)
        self.assertEqual(ruleset.select_records(0)[0]["rule_index"], 0)
        self.assertEqual(len(ruleset.select_records(slice(0, 2))), 2)
        self.assertEqual(ruleset.select_records([0, 1])[0]["rule_index"], 0)
        self.assertEqual(ruleset.select_records([ruleset.record(0)])[0]["rule_index"], 0)
        self.assertEqual(ruleset.select_records([ruleset[0]])[0]["is_black"], ruleset.record(0)["is_black"])
        opening_indexes = ruleset.legal_rule_indices(Turn.initial())
        opening_rules = ruleset.legal_rules(Turn.initial())
        self.assertEqual(
            [record["rule_index"] for record in ruleset.select_records(opening_rules)],
            opening_indexes,
        )
        with self.assertRaisesRegex(ValueError, "does not belong"):
            ruleset.select_records([Rule.from_masks(1, 0, 1 << 8, 1)])
        self.assertEqual(as_board(Board.initial()), Board.initial())
        self.assertEqual(as_board(Board.initial().as_tuple()), Board.initial())
        self.assertEqual(as_turn(Turn.initial()), Turn.initial())

    def test_ruleset_filtering_and_sorting(self):
        ruleset = Ruleset.american()
        records = ruleset.select_records(
            black_to_move="white",
            king=True,
            capture=True,
            sort_by=("promotion", "is_empty"),
            reverse=True,
        )
        indexes = ruleset.indices_for(
            black_to_move="white",
            king=True,
            capture=True,
            sort_by=("promotion", "is_empty"),
            reverse=True,
        )
        rules = ruleset.rules_for(
            black_to_move="white",
            king=True,
            capture=True,
            sort_by=("promotion", "is_empty"),
            reverse=True,
        )

        self.assertEqual(len(records), 42)
        self.assertTrue(all(not record["black_to_move"] for record in records))
        self.assertTrue(all(record["king"] and record["capture"] for record in records))
        self.assertEqual(
            [record["is_empty"] for record in records],
            sorted((record["is_empty"] for record in records), reverse=True),
        )
        self.assertEqual(indexes, [record["rule_index"] for record in records])
        self.assertEqual(rules, tuple(ruleset[index] for index in indexes))
        self.assertEqual(
            ruleset.indices_for(black_to_move="black", king=0, promotion=1, capture=0),
            ruleset.indices_for(black_to_move=1, king=0, promotion=1, capture=0),
        )

        with self.assertRaisesRegex(ValueError, "unknown rule sort field"):
            ruleset.rules_for(sort_by="not_a_field")

    def test_display_helpers_return_figures(self):
        ruleset = Ruleset.american()
        promotion_index = ruleset.indices_for(promotion=1)[0]
        board_fig, _board_ax = show_board(Board.initial(), size=1, show=False)
        turn_fig, _turn_ax = show_turn(Turn.initial(), size=1, show=False)
        board_method_fig, _board_method_ax = Board.initial().display(size=1, show=False)
        turn_method_fig, _turn_method_ax = Turn.initial().display(size=1, show=False)
        rule_method_fig, rule_method_axes = ruleset[0].display(size=1, show=False)
        indexed_rule_rows = ruleset.plot(rules=[0], size=1, show=False)
        display_rule_rows = ruleset.display(rules=[0], size=1, show=False)
        promotion_rule_rows = ruleset.plot(rules=[promotion_index], size=1, show=False)
        white_king_rule_rows = ruleset.plot(rules=ruleset.indices_for(black_to_move=0, king=1)[:1], size=1, show=False)
        filtered_rule_rows = ruleset.display(
            black_to_move="black",
            promotion=True,
            capture=True,
            sort_by=("king", "is_empty"),
            size=1,
            show=False,
        )
        dataframe_rule_rows = show_ruleset_rows(ruleset, rules=ruleset.to_dataframe().iloc[:1], size=1, show=False)
        direct_rule_rows = show_ruleset_rows([ruleset[0]], size=1, title="direct", show=False)

        self.assertIsNotNone(board_fig)
        self.assertIsNotNone(turn_fig)
        self.assertIsNotNone(board_method_fig)
        self.assertIsNotNone(turn_method_fig)
        self.assertIsNotNone(rule_method_fig)
        self.assertEqual(rule_method_axes.shape, (3,))
        self.assertEqual(len(indexed_rule_rows), 1)
        self.assertEqual(len(display_rule_rows), 1)
        self.assertEqual(len(promotion_rule_rows), 1)
        self.assertEqual(len(white_king_rule_rows), 1)
        self.assertEqual(len(filtered_rule_rows), 6)
        self.assertEqual(len(dataframe_rule_rows), 1)
        self.assertEqual(len(direct_rule_rows), 1)
        self.assertEqual(indexed_rule_rows[0][1].shape, (3,))
        self.assertIn("capture:", indexed_rule_rows[0][1][2].texts[0].get_text())
        self.assertIn("king required:", indexed_rule_rows[0][1][2].texts[0].get_text())
        untitled_fig, untitled_axes = _show_rule_row(ruleset.record(0), size=1, title=None)
        self.assertEqual(untitled_axes.shape, (3,))
        with self.assertRaises(ValueError):
            show_ruleset_rows(ruleset, rules=[], show=False)

        plt.close(board_fig)
        plt.close(turn_fig)
        plt.close(board_method_fig)
        plt.close(turn_method_fig)
        for fig, _axes in [
            (rule_method_fig, rule_method_axes),
            *indexed_rule_rows,
            *display_rule_rows,
            *promotion_rule_rows,
            *white_king_rule_rows,
            *filtered_rule_rows,
            *dataframe_rule_rows,
            *direct_rule_rows,
            (untitled_fig, untitled_axes),
        ]:
            plt.close(fig)

    def test_display_show_paths_are_exercised(self):
        ruleset = Ruleset.american()

        with patch("matplotlib.pyplot.show") as show:
            board_fig, _board_ax = Board.initial().display(size=1, show=True)
            show.assert_called_once()

        display = Mock()
        ipython_module = types.ModuleType("IPython")
        display_module = types.ModuleType("IPython.display")
        display_module.display = display
        ipython_module.display = display_module
        with patch.dict(sys.modules, {"IPython": ipython_module, "IPython.display": display_module}):
            rule_rows = ruleset.display(rules=[0], size=1, show=True)
        display.assert_called_once()

        with (
            patch("builtins.__import__", side_effect=_block_ipython_display_import),
            patch("matplotlib.pyplot.show") as show,
        ):
            fallback_fig, _fallback_ax = plt.subplots()
            _display_figure(fallback_fig)
            show.assert_called_once()

        plt.close(board_fig)
        plt.close(fallback_fig)
        for fig, _axes in rule_rows:
            plt.close(fig)


def _block_ipython_display_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "IPython.display":
        raise ImportError
    return _REAL_IMPORT(name, globals, locals, fromlist, level)


_REAL_IMPORT = __import__


if __name__ == "__main__":
    unittest.main()
