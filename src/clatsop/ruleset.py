"""Native 32-bit boards, turns, rules, and rulesets."""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from operator import index as integer_index
from typing import Any, SupportsIndex, TypeAlias, cast

from .encoding import MASK32, is_playable_square, square_from_mask32, square_mask32
from .state import (
    PackedState,
    StateKey,
    pack_state_values,
    pack_validated_state,
    quiet_ply_values,
    unpack_state_values,
    unpack_validated_state,
    validate_board_masks,
    validate_state_values,
)
from .state import (
    mask32 as _mask32,
)
from .state import (
    turn_bit as _turn_bit,
)

FLAG_BLACK_TO_MOVE = 1 << 0
FLAG_KING = 1 << 1
FLAG_PROMOTION = 1 << 2
FLAG_CAPTURE = 1 << 3

CAPTURE_AVAILABLE = 1 << 0
PROMOTION_AVAILABLE = 1 << 1
TERMINAL = 1 << 2

RuleKey: TypeAlias = tuple[int, int, int, int]
RulePath: TypeAlias = tuple[int, ...]
StateEdge: TypeAlias = tuple[RulePath, StateKey]
PackedStateEdge: TypeAlias = tuple[RulePath, PackedState]
RuleRecord: TypeAlias = dict[str, int]
RuleMetadataKey: TypeAlias = tuple[int, int, int, int]
RuntimeRow: TypeAlias = tuple[int, int, int, int, int, int]

RULE_COLUMNS = (
    "is_black",
    "is_white",
    "is_empty",
    "black_to_move",
    "king",
    "promotion",
    "capture",
)
RULE_FLAG_MASK = FLAG_BLACK_TO_MOVE | FLAG_KING | FLAG_PROMOTION | FLAG_CAPTURE

INITIAL_BLACK = 0x00000FFF
INITIAL_WHITE = 0xFFF00000


@dataclass(frozen=True, slots=True)
class Board:
    """A checkers board stored as black, white, and king 32-bit masks."""

    black: int
    white: int
    kings: int = 0

    def __post_init__(self) -> None:
        black, white, kings = validate_board_masks(self.black, self.white, self.kings)
        object.__setattr__(self, "black", black)
        object.__setattr__(self, "white", white)
        object.__setattr__(self, "kings", kings)

    @classmethod
    def initial(cls) -> Board:
        """Return the standard starting board."""

        return cls(INITIAL_BLACK, INITIAL_WHITE, 0)

    @classmethod
    def from_tuple(cls, values: object) -> Board:
        """Create a board from ``(black, white, kings)``."""

        values = tuple(values)  # type: ignore[arg-type]
        if len(values) != 3:
            raise ValueError("board tuple must have three fields")
        return cls(*validate_board_masks(*values))

    @property
    def occupied(self) -> int:
        """Return the occupied-square mask."""

        return self.black | self.white

    @property
    def empty(self) -> int:
        """Return the empty playable-square mask."""

        return (~self.occupied) & MASK32

    @property
    def key(self) -> tuple[int, int, int]:
        """Return a hashable native tuple key."""

        return self.as_tuple()

    @property
    def maps(self) -> dict[str, int]:
        """Return a plain dict of the board masks."""

        return self.as_dict()

    def as_tuple(self) -> tuple[int, int, int]:
        """Return ``(black, white, kings)``."""

        return self.black, self.white, self.kings

    def as_dict(self) -> dict[str, int]:
        """Return a plain dict of 32-bit masks."""

        return {"black": self.black, "white": self.white, "kings": self.kings}

    def display(
        self,
        size: float = 3,
        title: str | None = None,
        show: bool = True,
    ) -> tuple[Any, Any]:
        """Render this board with matplotlib and return ``(figure, axis)``."""

        from .display import show_board

        return show_board(self, size=size, title=title, show=show)


@dataclass(frozen=True, slots=True)
class Turn:
    """A board plus side to move and optional non-identity metadata."""

    board: Board
    black_to_move: int = 1
    metadata: tuple[tuple[object, object], ...] = field(default=(), compare=False, hash=False)

    def __post_init__(self) -> None:
        board = self.board if isinstance(self.board, Board) else Board.from_tuple(self.board)
        black_to_move = _turn_bit(self.black_to_move)
        metadata = _metadata_tuple(self.metadata)
        object.__setattr__(self, "board", board)
        object.__setattr__(self, "black_to_move", black_to_move)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def initial(cls) -> Turn:
        """Return the standard starting turn."""

        return cls(Board.initial(), 1)

    @classmethod
    def from_masks(
        cls,
        black: object,
        white: object,
        kings: object = 0,
        black_to_move: object = 1,
        metadata: object = (),
    ) -> Turn:
        """Create a turn from board masks and side to move."""

        board = Board(*validate_board_masks(black, white, kings))
        return cls(board, _turn_bit(black_to_move), _metadata_tuple(metadata))

    @classmethod
    def from_tuple(cls, values: object) -> Turn:
        """Create a turn from the four-field state key."""

        values = tuple(values)  # type: ignore[arg-type]
        if len(values) == 4:
            return cls.from_masks(*values)
        raise ValueError("turn tuple must have four fields")

    @classmethod
    def from_packed(cls, state: object, metadata: object = ()) -> Turn:
        """Create a validated turn from one packed 97-bit state integer."""

        return cls.from_masks(*unpack_state_values(state), metadata=metadata)

    @property
    def black(self) -> int:
        return self.board.black

    @property
    def white(self) -> int:
        return self.board.white

    @property
    def kings(self) -> int:
        return self.board.kings

    @property
    def occupied(self) -> int:
        return self.board.occupied

    @property
    def empty(self) -> int:
        return self.board.empty

    @property
    def side(self) -> str:
        """Return ``"black"`` or ``"white"`` for the side to move."""

        return "black" if self.black_to_move else "white"

    @property
    def key(self) -> StateKey:
        """Return the hashable game-state key, excluding analysis metadata."""

        return self.as_tuple()

    @property
    def packed(self) -> PackedState:
        """Return one compact 97-bit integer suitable for hash tables."""

        return pack_validated_state(*self.as_tuple())

    @property
    def maps(self) -> dict[str, object]:
        """Return a plain dictionary of board and turn masks."""

        return self.as_dict()

    def as_tuple(self) -> StateKey:
        """Return ``(black, white, kings, black_to_move)``."""

        return *self.board.as_tuple(), self.black_to_move

    def as_dict(self) -> dict[str, object]:
        """Return a plain dict for dataframe or ad hoc analysis use."""

        record: dict[str, object] = {**self.board.as_dict(), "black_to_move": self.black_to_move}
        if self.metadata:
            record["metadata"] = dict(self.metadata)
        return record

    def display(
        self,
        size: float = 3,
        title: str | None = None,
        show: bool = True,
    ) -> tuple[Any, Any]:
        """Render this turn with matplotlib and return ``(figure, axis)``."""

        from .display import show_turn

        return show_turn(self, size=size, title=title, show=show)


@dataclass(frozen=True, slots=True)
class Rule:
    """A primitive rule stored as three 32-bit masks and four packed flags."""

    is_black: int
    is_white: int
    is_empty: int
    flags: int

    def __post_init__(self) -> None:
        is_black = _mask32(self.is_black, "is_black")
        is_white = _mask32(self.is_white, "is_white")
        is_empty = _mask32(self.is_empty, "is_empty")
        flags = _integer(self.flags, "rule flags")
        if flags < 0 or flags & ~RULE_FLAG_MASK:
            raise ValueError("rule flags must fit in the four defined bits")
        if is_black & is_white or is_black & is_empty or is_white & is_empty:
            raise ValueError("rule masks must be disjoint")

        object.__setattr__(self, "is_black", is_black)
        object.__setattr__(self, "is_white", is_white)
        object.__setattr__(self, "is_empty", is_empty)
        object.__setattr__(self, "flags", flags)

        if self.mover.bit_count() != 1:
            raise ValueError("a rule must require exactly one moving piece")
        if self.is_empty.bit_count() != 1:
            raise ValueError("a rule must require exactly one destination square")
        opponent = self.is_white if self.black_to_move else self.is_black
        if opponent.bit_count() != self.capture:
            raise ValueError("the capture flag must match one required opponent")
        if self.promotion != _promotion_for_mask(self.black_to_move, self.king, self.is_empty):
            raise ValueError("the promotion flag does not match the destination")

    @classmethod
    def from_masks(
        cls,
        is_black: object,
        is_white: object,
        is_empty: object,
        black_to_move: object,
        king: object = 0,
        promotion: object | None = None,
        capture: object | None = None,
    ) -> Rule:
        """Create a rule from masks and unpacked metadata bits."""

        black_to_move = _turn_bit(black_to_move)
        king = _flag_bit(king, "king")
        is_black = _mask32(is_black, "is_black")
        is_white = _mask32(is_white, "is_white")
        is_empty = _mask32(is_empty, "is_empty")
        opponent = is_white if black_to_move else is_black
        capture = bool(opponent) if capture is None else _flag_bit(capture, "capture")
        if promotion is None:
            promotion = _promotion_for_mask(black_to_move, king, is_empty)
        else:
            promotion = _flag_bit(promotion, "promotion")
        return cls(
            is_black,
            is_white,
            is_empty,
            _pack_rule_flags(black_to_move, king, promotion, capture),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> Rule:
        """Create a rule from a compact flat record."""

        if "flags" in record:
            return cls(
                _mask32(record["is_black"], "is_black"),
                _mask32(record["is_white"], "is_white"),
                _mask32(record["is_empty"], "is_empty"),
                _integer(record["flags"], "rule flags"),
            )
        return cls.from_masks(
            record["is_black"],
            record["is_white"],
            record["is_empty"],
            record["black_to_move"],
            record.get("king", 0),
            record.get("promotion"),
            record.get("capture"),
        )

    def as_tuple(self) -> RuleKey:
        """Return the four stored values as a hashable tuple."""

        return self.is_black, self.is_white, self.is_empty, self.flags

    def as_dict(self) -> dict[str, int]:
        """Return an analysis-friendly record with unpacked metadata bits."""

        return {
            "is_black": self.is_black,
            "is_white": self.is_white,
            "is_empty": self.is_empty,
            "black_to_move": self.black_to_move,
            "king": self.king,
            "promotion": self.promotion,
            "capture": self.capture,
        }

    @property
    def black_to_move(self) -> int:
        """Return the packed side-to-move bit."""

        return int(bool(self.flags & FLAG_BLACK_TO_MOVE))

    @property
    def king(self) -> int:
        """Return the packed king-required bit."""

        return int(bool(self.flags & FLAG_KING))

    @property
    def promotion(self) -> int:
        """Return the packed promotion bit."""

        return int(bool(self.flags & FLAG_PROMOTION))

    @property
    def capture(self) -> int:
        """Return the packed capture bit."""

        return int(bool(self.flags & FLAG_CAPTURE))

    @property
    def mover(self) -> int:
        """Return the required mover-square mask."""

        return self.is_black if self.black_to_move else self.is_white

    def applies(self, turn: object) -> bool:
        """Return whether this rule's conditions match a turn."""

        turn = as_turn(turn)
        return _rule_applies_values(
            *turn.as_tuple(),
            self.is_black,
            self.is_white,
            self.is_empty,
            self.flags,
        )

    def apply(self, turn: object, switch_side: bool = True) -> Turn:
        """Apply this rule to a matching turn."""

        turn = as_turn(turn)
        if not self.applies(turn):
            raise ValueError("rule conditions are not satisfied by the turn")

        next_black, next_white, next_kings, next_side = _apply_rule_values(
            *turn.as_tuple(),
            self.is_black,
            self.is_white,
            self.is_empty,
            self.flags,
            switch_side,
        )
        return Turn(Board(next_black, next_white, next_kings), next_side, turn.metadata)

    def display(
        self,
        size: float = 3.2,
        title: str | None = None,
        show: bool = True,
    ) -> tuple[Any, Any]:
        """Render this primitive rule and return ``(figure, axes)``."""

        from .display import show_ruleset_rows

        return show_ruleset_rows([self], size=size, title=title, show=show)[0]


@dataclass(frozen=True, slots=True)
class Transition:
    """A completed legal turn represented by primitive rule indexes and its result."""

    rule_indices: RulePath
    turn: Turn

    def __post_init__(self) -> None:
        rule_indices = tuple(_integer(index, "rule index") for index in self.rule_indices)
        if not rule_indices:
            raise ValueError("a transition must contain at least one primitive rule")
        object.__setattr__(self, "rule_indices", rule_indices)
        object.__setattr__(self, "turn", as_turn(self.turn))

    @property
    def key(self) -> tuple[RulePath, StateKey]:
        """Return a hashable path/result key."""

        return self.rule_indices, self.turn.key

    def as_dict(self) -> dict[str, object]:
        """Return a plain transition record."""

        return {"rule_indices": self.rule_indices, "turn": self.turn.as_dict()}


class Ruleset:
    """Compact primitive rules with bitwise runtime indexes and Python views."""

    __slots__ = (
        "_is_black",
        "_is_white",
        "_is_empty",
        "_flags",
        "_candidate_rows",
        "_rule_indices",
        "_rules_by_side",
        "_rules_by_metadata",
    )

    def __init__(self, rules: Iterable[object]):
        rule_tuple = tuple(_coerce_rule(rule) for rule in rules)
        rule_indices: dict[RuleKey, int] = {}
        for index, rule in enumerate(rule_tuple):
            key = rule.as_tuple()
            if key in rule_indices:
                raise ValueError(f"duplicate rule at indexes {rule_indices[key]} and {index}")
            rule_indices[key] = index
        self._is_black = array("I", (rule.is_black for rule in rule_tuple))
        self._is_white = array("I", (rule.is_white for rule in rule_tuple))
        self._is_empty = array("I", (rule.is_empty for rule in rule_tuple))
        self._flags = array("B", (rule.flags for rule in rule_tuple))
        runtime_rows = tuple(
            (
                index,
                rule.is_black,
                rule.is_white,
                rule.is_empty,
                rule.flags,
                rule.mover,
            )
            for index, rule in enumerate(rule_tuple)
        )
        self._candidate_rows = _runtime_candidate_rows(runtime_rows)
        self._rule_indices = rule_indices
        self._rules_by_side = _rules_by_side(rule_tuple)
        self._rules_by_metadata = _rules_by_metadata(rule_tuple)

    @property
    def rules(self) -> tuple[Rule, ...]:
        """Return the immutable primitive-rule tuple."""

        return tuple(self)

    @property
    def records(self) -> tuple[RuleRecord, ...]:
        """Return independent plain dictionaries for all primitive rules."""

        return tuple(self.record(index) for index in range(len(self)))

    @property
    def record_map(self) -> dict[int, RuleRecord]:
        """Return an independent ``{rule_index: record}`` dictionary."""

        return {index: self.record(index) for index in range(len(self))}

    @property
    def rule_set(self) -> set[RuleKey]:
        """Return an independent set of hashable primitive-rule tuples."""

        return {
            (
                self._is_black[index],
                self._is_white[index],
                self._is_empty[index],
                self._flags[index],
            )
            for index in range(len(self))
        }

    @property
    def buffers(self) -> dict[str, memoryview]:
        """Return immutable typed snapshots of the packed runtime columns."""

        return {
            "is_black": memoryview(self._is_black.tobytes()).cast("I"),
            "is_white": memoryview(self._is_white.tobytes()).cast("I"),
            "is_empty": memoryview(self._is_empty.tobytes()).cast("I"),
            "flags": memoryview(self._flags.tobytes()).cast("B"),
        }

    @property
    def rules_by_side(self) -> dict[int, tuple[int, ...]]:
        """Return an independent side-to-rule-index mapping."""

        return dict(self._rules_by_side)

    @property
    def rules_by_metadata(self) -> dict[RuleMetadataKey, tuple[int, ...]]:
        """Return an independent metadata-to-rule-index mapping."""

        return dict(self._rules_by_metadata)

    @classmethod
    def american(cls, side: object = "both") -> Ruleset:
        """Return the standard American checkers primitive ruleset."""

        if side is None or str(side).lower() in ("both", "all"):
            return american_ruleset()
        return cls.from_board_geometry(side=side)

    @classmethod
    def from_board_geometry(cls, side: object = "both") -> Ruleset:
        """Build rules directly from the 32 playable-square board geometry."""

        rules = []
        for black_to_move in _runtime_sides(side):
            for template in _compact_move_templates():
                if _template_requires_king(black_to_move, template):
                    rules.append(_rule_from_template(template, black_to_move, king=1))
                else:
                    rules.append(_rule_from_template(template, black_to_move, king=0))
                    if _template_promotes(black_to_move, template):
                        rules.append(_rule_from_template(template, black_to_move, king=1))
        return cls(rules)

    @classmethod
    def from_records(cls, records: object) -> Ruleset:
        """Build a ruleset from flat records, rule objects, or a dataframe."""

        return cls(_records_from_any(records))

    def __len__(self) -> int:
        return len(self._flags)

    def __iter__(self) -> Iterator[Rule]:
        return (self[index] for index in range(len(self)))

    def __getitem__(self, rule_index: object) -> Rule:
        index = self._check_index(rule_index)
        return Rule(
            self._is_black[index],
            self._is_white[index],
            self._is_empty[index],
            self._flags[index],
        )

    def record(self, rule_index: object) -> RuleRecord:
        """Return one rule as a plain dictionary."""

        index = self._check_index(rule_index)
        return _indexed_record(index, self[index])

    def to_dataframe(self) -> Any:
        """Return three stored masks and four unpacked metadata bits."""

        import pandas as pd

        df = pd.DataFrame.from_records(self.records, columns=("rule_index", *RULE_COLUMNS))
        return df.set_index("rule_index").rename_axis(None)

    def select_records(
        self,
        rules: object | None = None,
        *,
        black_to_move: object | None = None,
        king: object | None = None,
        promotion: object | None = None,
        capture: object | None = None,
        sort_by: str | Iterable[str] | None = None,
        reverse: bool = False,
    ) -> list[RuleRecord]:
        """Return filtered and sorted rule records from common selection inputs."""

        black_to_move, king, promotion, capture = _rule_filter_values(
            black_to_move,
            king,
            promotion,
            capture,
        )
        records = None
        if rules is None:
            records = [dict(record) for record in self.records]
        elif isinstance(rules, int):
            records = [self.record(rules)]
        elif isinstance(rules, slice):
            records = [self.record(index) for index in range(len(self))[rules]]
        elif hasattr(rules, "to_dict") and hasattr(rules, "columns"):
            rows = []
            dataframe = cast(Any, rules)
            for index, record in dataframe.to_dict("index").items():
                record = dict(record)
                record.setdefault("rule_index", index)
                rows.append(record)
            records = rows
        else:
            values = list(cast(Iterable[object], rules))
            if all(isinstance(value, int) for value in values):
                records = [self.record(index) for index in values]
            else:
                records = []
                for value in values:
                    if isinstance(value, Rule):
                        key = value.as_tuple()
                        if key not in self._rule_indices:
                            raise ValueError("selected rule does not belong to this ruleset")
                        records.append(_indexed_record(self._rule_indices[key], value))
                    else:
                        records.append(dict(cast(Mapping[str, int], value)))

        return _query_rule_records(
            records,
            black_to_move=black_to_move,
            king=king,
            promotion=promotion,
            capture=capture,
            sort_by=sort_by,
            reverse=reverse,
        )

    def plot(
        self,
        rules: object | None = None,
        size: float = 3.2,
        title: str | None = None,
        show: bool = True,
        *,
        black_to_move: object | None = None,
        king: object | None = None,
        promotion: object | None = None,
        capture: object | None = None,
        sort_by: str | Iterable[str] | None = None,
        reverse: bool = False,
    ) -> list[tuple[Any, Any]]:
        """Plot selected rules with condition, effect, and summary boards."""

        from .display import show_ruleset_rows

        selected = self.select_records(
            rules,
            black_to_move=black_to_move,
            king=king,
            promotion=promotion,
            capture=capture,
            sort_by=sort_by,
            reverse=reverse,
        )
        return show_ruleset_rows(selected, size=size, title=title, show=show)

    def display(
        self,
        rules: object | None = None,
        size: float = 3.2,
        title: str | None = None,
        show: bool = True,
        *,
        black_to_move: object | None = None,
        king: object | None = None,
        promotion: object | None = None,
        capture: object | None = None,
        sort_by: str | Iterable[str] | None = None,
        reverse: bool = False,
    ) -> list[tuple[Any, Any]]:
        """Render selected rules with condition, effect, and summary boards."""

        return self.plot(
            rules=rules,
            size=size,
            title=title,
            show=show,
            black_to_move=black_to_move,
            king=king,
            promotion=promotion,
            capture=capture,
            sort_by=sort_by,
            reverse=reverse,
        )

    def indices_for(
        self,
        black_to_move: object | None = None,
        king: object | None = None,
        promotion: object | None = None,
        capture: object | None = None,
        *,
        sort_by: str | Iterable[str] | None = None,
        reverse: bool = False,
    ) -> list[int]:
        """Return rule indexes matching metadata filters."""

        black_to_move, king, promotion, capture = _rule_filter_values(
            black_to_move,
            king,
            promotion,
            capture,
        )
        if all(value is not None for value in (black_to_move, king, promotion, capture)):
            key = (
                black_to_move,
                king,
                promotion,
                capture,
            )
            indexes = list(self._rules_by_metadata.get(key, ()))
        else:
            indexes = []
            for key, key_indexes in self._rules_by_metadata.items():
                if all(
                    requested is None or value == requested
                    for value, requested in zip(key, (black_to_move, king, promotion, capture), strict=True)
                ):
                    indexes.extend(key_indexes)
            indexes.sort()

        if sort_by is None:
            return indexes
        records = _sort_rule_records([self.record(index) for index in indexes], sort_by, reverse)
        return [record["rule_index"] for record in records]

    def rules_for(
        self,
        *,
        black_to_move: object | None = None,
        king: object | None = None,
        promotion: object | None = None,
        capture: object | None = None,
        sort_by: str | Iterable[str] | None = None,
        reverse: bool = False,
    ) -> tuple[Rule, ...]:
        """Return rule objects matching metadata filters in the requested order."""

        indexes = self.indices_for(
            black_to_move,
            king,
            promotion,
            capture,
            sort_by=sort_by,
            reverse=reverse,
        )
        return tuple(self[index] for index in indexes)

    def matching_rule_indices(
        self,
        turn: object,
        *,
        capture: object | None = None,
        allow_promotions: object = True,
        from_mask: object | None = None,
    ) -> list[int]:
        """Return bitwise-matched rule indexes under runtime policy filters."""

        turn = as_turn(turn)
        capture = _optional_flag_bit(capture, "capture")
        capture_mode = None if capture is None else bool(capture)
        allow_promotions = bool(_flag_bit(allow_promotions, "allow_promotions"))
        if from_mask is not None:
            from_mask = _mask32(from_mask, "from_mask")
        return self._matching_indices(
            turn.black,
            turn.white,
            turn.kings,
            turn.black_to_move,
            capture_mode,
            allow_promotions,
            from_mask,
        )

    def matching_rules(
        self,
        turn: object,
        *,
        capture: object | None = None,
        allow_promotions: object = True,
        from_mask: object | None = None,
    ) -> tuple[Rule, ...]:
        """Return matching rule objects."""

        indexes = self.matching_rule_indices(
            turn,
            capture=capture,
            allow_promotions=allow_promotions,
            from_mask=from_mask,
        )
        return tuple(self[index] for index in indexes)

    def legal_rule_indices(self, turn: object, from_mask: object | None = None) -> list[int]:
        """Return legal primitive rule indexes, applying mandatory capture."""

        turn = as_turn(turn)
        captures = self._matching_indices(
            turn.black,
            turn.white,
            turn.kings,
            turn.black_to_move,
            True,
            True,
            None,
        )
        legal = captures or self._matching_indices(
            turn.black,
            turn.white,
            turn.kings,
            turn.black_to_move,
            False,
            True,
            None,
        )
        if from_mask is None:
            return legal
        from_mask = _mask32(from_mask, "from_mask")
        return [index for index in legal if self._mover(index) == from_mask]

    def legal_rules(self, turn: object, from_mask: object | None = None) -> tuple[Rule, ...]:
        """Return legal primitive rule objects."""

        return tuple(self[index] for index in self.legal_rule_indices(turn, from_mask=from_mask))

    def has_capture(self, turn: object, from_mask: object | None = None) -> bool:
        """Return whether the side to move has an immediately legal capture."""

        black, white, kings, black_to_move = _turn_values(turn)
        if from_mask is not None:
            from_mask = _mask32(from_mask, "from_mask")
        return self._has_matching(
            black,
            white,
            kings,
            black_to_move,
            True,
            True,
            from_mask,
        )

    def has_promotion(self, turn: object) -> bool:
        """Return whether any legal first-step move promotes a man."""

        return any(self._flags[index] & FLAG_PROMOTION for index in self.legal_rule_indices(turn))

    def apply_rule(
        self,
        turn: object,
        rule_index: object,
        switch_side: bool = True,
    ) -> Turn:
        """Apply one satisfied primitive rule to a turn."""

        turn = as_turn(turn)
        index = self._check_index(rule_index)
        values = turn.as_tuple()
        if not self._index_applies(*values, index):
            raise ValueError("rule conditions are not satisfied by the turn")
        black, white, kings, black_to_move = self._apply_index(*values, index, switch_side)
        return Turn.from_masks(black, white, kings, black_to_move, turn.metadata)

    def apply_key(
        self,
        turn: object,
        rule_index: object,
        switch_side: bool = True,
    ) -> StateKey:
        """Apply one rule and return a native four-integer state key."""

        values = _turn_values(turn)
        index = self._check_index(rule_index)
        if not self._index_applies(*values, index):
            raise ValueError("rule conditions are not satisfied by the turn")
        return self._apply_index(*values, index, switch_side)

    def apply_packed(
        self,
        state: object,
        rule_index: object,
        switch_side: bool = True,
    ) -> PackedState:
        """Apply one rule to a packed state and return a packed successor."""

        values = unpack_state_values(state)
        index = self._check_index(rule_index)
        if not self._index_applies(*values, index):
            raise ValueError("rule conditions are not satisfied by the turn")
        return pack_validated_state(*self._apply_index(*values, index, switch_side))

    def expand_keys(
        self,
        turn: object,
        *,
        allow_captures: bool = True,
        allow_promotions: bool = True,
    ) -> tuple[tuple[StateKey, ...], int]:
        """Return legal successor keys and packed capture/promotion/terminal status."""

        values = _turn_values(turn)
        successors, status = self._expand_values(values, allow_captures, allow_promotions)
        return tuple(successors), status

    def expand_packed(
        self,
        state: object,
        *,
        allow_captures: bool = True,
        allow_promotions: bool = True,
    ) -> tuple[tuple[PackedState, ...], int]:
        """Expand one packed state without allocating board or turn objects."""

        values = unpack_state_values(state)
        successors, status = self._expand_values(values, allow_captures, allow_promotions)
        return tuple(pack_validated_state(*successor) for successor in successors), status

    def expand_packed_unchecked(
        self,
        state: PackedState,
        *,
        allow_captures: bool = True,
        allow_promotions: bool = True,
    ) -> tuple[tuple[PackedState, ...], int]:
        """Expand a trusted packed state without repeating invariant checks."""

        values = unpack_validated_state(state)
        successors, status = self._expand_values(values, allow_captures, allow_promotions)
        return tuple(pack_validated_state(*successor) for successor in successors), status

    def expand_edges(
        self,
        turn: object,
        *,
        allow_captures: bool = True,
        allow_promotions: bool = True,
    ) -> tuple[tuple[StateEdge, ...], int]:
        """Return completed rule paths paired with native successor keys."""

        values = _turn_values(turn)
        edges, status = self._expand_value_edges(values, allow_captures, allow_promotions)
        return tuple(edges), status

    def expand_packed_edges(
        self,
        state: object,
        *,
        allow_captures: bool = True,
        allow_promotions: bool = True,
    ) -> tuple[tuple[PackedStateEdge, ...], int]:
        """Return completed rule paths paired with packed successor states."""

        values = unpack_state_values(state)
        edges, status = self._expand_value_edges(values, allow_captures, allow_promotions)
        return tuple((path, pack_validated_state(*successor)) for path, successor in edges), status

    def _expand_values(
        self,
        values: StateKey,
        allow_captures: bool,
        allow_promotions: bool,
    ) -> tuple[list[StateKey], int]:
        black, white, kings, black_to_move = values
        captures = self._matching_indices(
            black,
            white,
            kings,
            black_to_move,
            True,
            True,
            None,
        )
        if captures:
            status = CAPTURE_AVAILABLE
            if any(self._flags[index] & FLAG_PROMOTION for index in captures):
                status |= PROMOTION_AVAILABLE
            if not allow_captures:
                return [], status
            successors, chain_status = self._capture_key_successors(
                values,
                captures,
                allow_promotions,
            )
            return successors, status | chain_status

        quiet = self._matching_indices(
            black,
            white,
            kings,
            black_to_move,
            False,
            True,
            None,
        )
        if not quiet:
            return [], TERMINAL

        status = 0
        successors = []
        for index in quiet:
            if self._flags[index] & FLAG_PROMOTION:
                status |= PROMOTION_AVAILABLE
                if not allow_promotions:
                    continue
            successors.append(self._apply_index(*values, index, True))
        return successors, status

    def _expand_value_edges(
        self,
        values: StateKey,
        allow_captures: bool,
        allow_promotions: bool,
    ) -> tuple[list[StateEdge], int]:
        black, white, kings, black_to_move = values
        captures = self._matching_indices(
            black,
            white,
            kings,
            black_to_move,
            True,
            True,
            None,
        )
        if captures:
            status = CAPTURE_AVAILABLE
            if any(self._flags[index] & FLAG_PROMOTION for index in captures):
                status |= PROMOTION_AVAILABLE
            if not allow_captures:
                return [], status
            edges, chain_status = self._capture_key_edges(
                values,
                captures,
                allow_promotions,
                (),
            )
            return edges, status | chain_status

        quiet = self._matching_indices(
            black,
            white,
            kings,
            black_to_move,
            False,
            True,
            None,
        )
        if not quiet:
            return [], TERMINAL

        status = 0
        edges = []
        for index in quiet:
            if self._flags[index] & FLAG_PROMOTION:
                status |= PROMOTION_AVAILABLE
                if not allow_promotions:
                    continue
            edges.append(((index,), self._apply_index(*values, index, True)))
        return edges, status

    def legal_transitions(self, turn: object) -> tuple[Transition, ...]:
        """Return completed legal turns, expanding all mandatory capture chains."""

        turn = as_turn(turn)
        indexes = self.legal_rule_indices(turn)
        if not indexes:
            return ()
        if not self._flags[indexes[0]] & FLAG_CAPTURE:
            return tuple(Transition((index,), self.apply_rule(turn, index)) for index in indexes)

        transitions = []
        for index in indexes:
            primitive_turn = self.apply_rule(turn, index, switch_side=False)
            if self._flags[index] & FLAG_PROMOTION:
                transitions.append(Transition((index,), _finish_turn(primitive_turn)))
            else:
                transitions.extend(self._capture_transitions(primitive_turn, self._is_empty[index], (index,)))
        return tuple(transitions)

    def successors(self, turn: object) -> tuple[Turn, ...]:
        """Return successor turns after complete legal moves."""

        return tuple(transition.turn for transition in self.legal_transitions(turn))

    def successor_map(self, turn: object) -> dict[RulePath, Turn]:
        """Return ``{rule_index_path: successor_turn}`` for complete legal moves."""

        return {transition.rule_indices: transition.turn for transition in self.legal_transitions(turn)}

    def _capture_transitions(self, turn: Turn, from_mask: int, path: RulePath) -> list[Transition]:
        indexes = self.matching_rule_indices(turn, capture=True, from_mask=from_mask)
        if not indexes:
            return [Transition(path, _finish_turn(turn))]

        transitions = []
        for index in indexes:
            primitive_turn = self.apply_rule(turn, index, switch_side=False)
            next_path = (*path, index)
            if self._flags[index] & FLAG_PROMOTION:
                transitions.append(Transition(next_path, _finish_turn(primitive_turn)))
            else:
                transitions.extend(self._capture_transitions(primitive_turn, self._is_empty[index], next_path))
        return transitions

    def _matching_indices(
        self,
        black: int,
        white: int,
        kings: int,
        black_to_move: int,
        capture: bool | None,
        allow_promotions: bool,
        from_mask: int | None,
    ) -> list[int]:
        empty = (~(black | white)) & MASK32
        moving_kings = kings & (black if black_to_move else white)
        rows = self._candidate_rows[
            _runtime_candidate_key(
                black_to_move,
                bool(moving_kings),
                allow_promotions,
                capture,
            )
        ]
        matches = []
        for index, is_black, is_white, is_empty, flags, mover in rows:
            if from_mask is not None and mover != from_mask:
                continue
            if black & is_black != is_black or white & is_white != is_white or empty & is_empty != is_empty:
                continue
            mover_is_king = kings & mover
            if flags & FLAG_KING and not mover_is_king:
                continue
            if flags & FLAG_PROMOTION and mover_is_king:
                continue
            matches.append(index)
        return matches

    def _has_matching(
        self,
        black: int,
        white: int,
        kings: int,
        black_to_move: int,
        capture: bool | None,
        allow_promotions: bool,
        from_mask: int | None,
    ) -> bool:
        empty = (~(black | white)) & MASK32
        moving_kings = kings & (black if black_to_move else white)
        rows = self._candidate_rows[
            _runtime_candidate_key(
                black_to_move,
                bool(moving_kings),
                allow_promotions,
                capture,
            )
        ]
        for _index, is_black, is_white, is_empty, flags, mover in rows:
            if from_mask is not None and mover != from_mask:
                continue
            if black & is_black != is_black or white & is_white != is_white or empty & is_empty != is_empty:
                continue
            mover_is_king = kings & mover
            if flags & FLAG_KING and not mover_is_king:
                continue
            if flags & FLAG_PROMOTION and mover_is_king:
                continue
            return True
        return False

    def _index_applies(
        self,
        black: int,
        white: int,
        kings: int,
        black_to_move: int,
        index: int,
    ) -> bool:
        return _rule_applies_values(
            black,
            white,
            kings,
            black_to_move,
            self._is_black[index],
            self._is_white[index],
            self._is_empty[index],
            self._flags[index],
        )

    def _apply_index(
        self,
        black: int,
        white: int,
        kings: int,
        black_to_move: int,
        index: int,
        switch_side: bool,
    ) -> StateKey:
        is_black = self._is_black[index]
        is_white = self._is_white[index]
        destination = self._is_empty[index]
        cleared = is_black | is_white
        if black_to_move:
            next_black = (black & ~cleared) | destination
            next_white = white & ~cleared
            mover = is_black
        else:
            next_black = black & ~cleared
            next_white = (white & ~cleared) | destination
            mover = is_white
        next_kings = kings & ~cleared
        if kings & mover or self._flags[index] & FLAG_PROMOTION:
            next_kings |= destination
        next_side = 1 - black_to_move if switch_side else black_to_move
        return next_black, next_white, next_kings, next_side

    def _capture_key_successors(
        self,
        values: StateKey,
        indexes: list[int],
        allow_promotions: bool,
    ) -> tuple[list[StateKey], int]:
        black_to_move = values[3]
        successors = []
        status = 0
        for index in indexes:
            primitive = self._apply_index(*values, index, False)
            if self._flags[index] & FLAG_PROMOTION:
                status |= PROMOTION_AVAILABLE
                if allow_promotions:
                    successors.append((*primitive[:3], 1 - black_to_move))
                continue

            next_captures = self._matching_indices(
                *primitive,
                True,
                True,
                self._is_empty[index],
            )
            if next_captures:
                nested, nested_status = self._capture_key_successors(
                    primitive,
                    next_captures,
                    allow_promotions,
                )
                successors.extend(nested)
                status |= nested_status
            else:
                successors.append((*primitive[:3], 1 - black_to_move))
        return successors, status

    def _capture_key_edges(
        self,
        values: StateKey,
        indexes: list[int],
        allow_promotions: bool,
        path: RulePath,
    ) -> tuple[list[StateEdge], int]:
        black_to_move = values[3]
        edges = []
        status = 0
        for index in indexes:
            primitive = self._apply_index(*values, index, False)
            next_path = (*path, index)
            if self._flags[index] & FLAG_PROMOTION:
                status |= PROMOTION_AVAILABLE
                if allow_promotions:
                    edges.append((next_path, (*primitive[:3], 1 - black_to_move)))
                continue

            next_captures = self._matching_indices(
                *primitive,
                True,
                True,
                self._is_empty[index],
            )
            if next_captures:
                nested, nested_status = self._capture_key_edges(
                    primitive,
                    next_captures,
                    allow_promotions,
                    next_path,
                )
                edges.extend(nested)
                status |= nested_status
            else:
                edges.append((next_path, (*primitive[:3], 1 - black_to_move)))
        return edges, status

    def _mover(self, index: int) -> int:
        return self._is_black[index] if self._flags[index] & FLAG_BLACK_TO_MOVE else self._is_white[index]

    def _check_index(self, rule_index: object) -> int:
        index = _integer(rule_index, "rule index")
        if not (0 <= index < len(self)):
            raise IndexError(f"rule index out of range: {rule_index!r}")
        return index


def _rule_applies_values(
    black: int,
    white: int,
    kings: int,
    black_to_move: int,
    is_black: int,
    is_white: int,
    is_empty: int,
    flags: int,
) -> bool:
    if bool(flags & FLAG_BLACK_TO_MOVE) != bool(black_to_move):
        return False
    if black & is_black != is_black or white & is_white != is_white:
        return False
    if (~(black | white)) & MASK32 & is_empty != is_empty:
        return False
    mover = is_black if black_to_move else is_white
    mover_is_king = bool(kings & mover)
    if flags & FLAG_KING:
        return mover_is_king
    if flags & FLAG_PROMOTION:
        return not mover_is_king
    return True


def _apply_rule_values(
    black: int,
    white: int,
    kings: int,
    black_to_move: int,
    is_black: int,
    is_white: int,
    destination: int,
    flags: int,
    switch_side: bool,
) -> StateKey:
    cleared = is_black | is_white
    if black_to_move:
        next_black = (black & ~cleared) | destination
        next_white = white & ~cleared
        mover = is_black
    else:
        next_black = black & ~cleared
        next_white = (white & ~cleared) | destination
        mover = is_white
    next_kings = kings & ~cleared
    if kings & mover or flags & FLAG_PROMOTION:
        next_kings |= destination
    next_side = 1 - black_to_move if switch_side else black_to_move
    return next_black, next_white, next_kings, next_side


def as_board(board: object) -> Board:
    """Coerce a board-like object into :class:`Board`."""

    if isinstance(board, Board):
        return board
    if isinstance(board, Mapping):
        return Board(board["black"], board["white"], board.get("kings", 0))
    return Board.from_tuple(board)


def as_turn(turn: object) -> Turn:
    """Coerce a turn-like object into :class:`Turn`."""

    if isinstance(turn, Turn):
        return turn
    if isinstance(turn, Mapping):
        return Turn.from_masks(
            turn["black"],
            turn["white"],
            turn.get("kings", 0),
            turn.get("black_to_move", 1),
            turn.get("metadata", ()),
        )
    return Turn.from_tuple(turn)


def pack_state(turn: object) -> PackedState:
    """Validate and pack a turn-like value into one 97-bit integer."""

    return pack_state_values(*_turn_values(turn))


def unpack_state(state: object) -> StateKey:
    """Validate and unpack one 97-bit integer into a native state key."""

    return unpack_state_values(state)


def quiet_ply(turn: object) -> int:
    """Return a no-capture, no-promotion state's unique layer from the start."""

    return quiet_ply_values(*_turn_values(turn))


def american_ruleset() -> Ruleset:
    """Return the shared default American checkers ruleset."""

    global _DEFAULT_AMERICAN_RULESET
    if _DEFAULT_AMERICAN_RULESET is None:
        _DEFAULT_AMERICAN_RULESET = Ruleset.from_board_geometry()
    return _DEFAULT_AMERICAN_RULESET


def _indexed_record(index, rule):
    return {"rule_index": index, **rule.as_dict()}


def _coerce_rule(rule: object) -> Rule:
    if isinstance(rule, Rule):
        return rule
    return Rule.from_record(cast(Mapping[str, object], rule))


def _records_from_any(records: object) -> Iterable[object]:
    if isinstance(records, Ruleset):
        return records.rules
    if isinstance(records, Rule):
        return (records,)
    if isinstance(records, Mapping):
        return (records,)
    if hasattr(records, "to_dict") and hasattr(records, "columns"):
        return tuple(cast(Any, records).to_dict("records"))
    if hasattr(records, "to_dict") and not isinstance(records, Mapping):
        return (cast(Any, records).to_dict(),)
    return cast(Iterable[object], records)


def _rules_by_side(rules):
    indexes = {0: [], 1: []}
    for index, rule in enumerate(rules):
        indexes[rule.black_to_move].append(index)
    return {side: tuple(side_indexes) for side, side_indexes in indexes.items()}


def _rules_by_metadata(rules):
    indexes = {}
    for index, rule in enumerate(rules):
        key = (
            rule.black_to_move,
            rule.king,
            rule.promotion,
            rule.capture,
        )
        indexes.setdefault(key, []).append(index)
    return {key: tuple(value) for key, value in indexes.items()}


def _runtime_candidate_rows(
    runtime_rows: tuple[RuntimeRow, ...],
) -> tuple[tuple[RuntimeRow, ...], ...]:
    candidates: list[tuple[RuntimeRow, ...]] = [()] * 24
    for black_to_move in (0, 1):
        for has_kings in (False, True):
            for allow_promotions in (False, True):
                for capture in (None, False, True):
                    rows: list[RuntimeRow] = []
                    for row in runtime_rows:
                        flags = row[4]
                        if bool(flags & FLAG_BLACK_TO_MOVE) != bool(black_to_move):
                            continue
                        if not has_kings and flags & FLAG_KING:
                            continue
                        if not allow_promotions and flags & FLAG_PROMOTION:
                            continue
                        if capture is not None and bool(flags & FLAG_CAPTURE) != capture:
                            continue
                        rows.append(row)
                    candidates[
                        _runtime_candidate_key(
                            black_to_move,
                            has_kings,
                            allow_promotions,
                            capture,
                        )
                    ] = tuple(rows)
    return tuple(candidates)


def _runtime_candidate_key(
    black_to_move: int,
    has_kings: bool,
    allow_promotions: bool,
    capture: bool | None,
) -> int:
    if capture is None:
        capture_mode = 0
    elif capture:
        capture_mode = 2
    else:
        capture_mode = 1
    return int(bool(black_to_move)) | int(bool(has_kings)) << 1 | int(bool(allow_promotions)) << 2 | capture_mode << 3


def _query_rule_records(
    records,
    *,
    black_to_move=None,
    king=None,
    promotion=None,
    capture=None,
    sort_by=None,
    reverse=False,
):
    if black_to_move is not None:
        black_to_move = _turn_bit(black_to_move)

    selected = []
    for fallback_index, value in enumerate(records):
        record = dict(value)
        record.setdefault("rule_index", fallback_index)
        if (
            _match_flag(record["black_to_move"], black_to_move)
            and _match_flag(record["king"], king)
            and _match_flag(record["promotion"], promotion)
            and _match_flag(record["capture"], capture)
        ):
            selected.append(record)
    return _sort_rule_records(selected, sort_by, reverse)


def _sort_rule_records(records, sort_by, reverse=False):
    if sort_by is None:
        return records
    fields = (sort_by,) if isinstance(sort_by, str) else tuple(sort_by)
    invalid = set(fields) - {"rule_index", *RULE_COLUMNS}
    if invalid:
        names = ", ".join(sorted(invalid))
        raise ValueError(f"unknown rule sort field(s): {names}")
    return sorted(records, key=lambda record: tuple(record[field] for field in fields), reverse=bool(reverse))


def _match_flag(value, requested):
    return requested is None or value == requested


def _rule_filter_values(black_to_move, king, promotion, capture):
    if black_to_move is not None:
        black_to_move = _turn_bit(black_to_move)
    return (
        black_to_move,
        _optional_flag_bit(king, "king"),
        _optional_flag_bit(promotion, "promotion"),
        _optional_flag_bit(capture, "capture"),
    )


def _optional_flag_bit(value, name):
    return None if value is None else _flag_bit(value, name)


_DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _runtime_sides(side):
    if side is None or str(side).lower() in ("both", "all"):
        return (1, 0)
    return (_turn_bit(side),)


def _compact_move_templates():
    templates = []
    for row in range(8):
        for col in range(8):
            if not is_playable_square(row, col):
                continue
            from_mask = square_mask32(row, col)
            for dr, dc in _DIRECTIONS:
                to_row = row + dr
                to_col = col + dc
                if is_playable_square(to_row, to_col):
                    templates.append((from_mask, square_mask32(to_row, to_col), 0))

                jump_row = row + 2 * dr
                jump_col = col + 2 * dc
                captured_row = row + dr
                captured_col = col + dc
                if is_playable_square(jump_row, jump_col):
                    templates.append(
                        (
                            from_mask,
                            square_mask32(jump_row, jump_col),
                            square_mask32(captured_row, captured_col),
                        )
                    )
    return templates


def _template_requires_king(black_to_move, template):
    from_row, _ = square_from_mask32(template[0])
    to_row, _ = square_from_mask32(template[1])
    dr = to_row - from_row
    return (black_to_move and dr < 0) or (not black_to_move and dr > 0)


def _template_promotes(black_to_move, template):
    destination_row, _ = square_from_mask32(template[1])
    return _promotion_row(black_to_move, destination_row)


def _rule_from_template(template, black_to_move, king):
    from_mask, to_mask, captured_mask = template
    to_row, _ = square_from_mask32(to_mask)
    promotion = not king and _promotion_row(black_to_move, to_row)

    if black_to_move:
        is_black, is_white = from_mask, captured_mask
    else:
        is_black, is_white = captured_mask, from_mask
    return Rule.from_masks(
        is_black,
        is_white,
        to_mask,
        black_to_move,
        king,
        promotion,
        bool(captured_mask),
    )


def _pack_rule_flags(black_to_move: int, king: int, promotion: int, capture: int) -> int:
    return black_to_move * FLAG_BLACK_TO_MOVE | king * FLAG_KING | promotion * FLAG_PROMOTION | capture * FLAG_CAPTURE


def _promotion_for_mask(black_to_move: int, king: int, destination: object) -> int:
    destination = _mask32(destination, "is_empty")
    promotion_rank = 0xF0000000 if black_to_move else 0x0000000F
    return int(not king and bool(destination & promotion_rank))


def _promotion_row(black_to_move: int, row: int) -> bool:
    return (black_to_move and row == 7) or (not black_to_move and row == 0)


def _integer(value: object, name: str) -> int:
    try:
        return integer_index(cast(SupportsIndex, value))
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def _flag_bit(value: object, name: str) -> int:
    value = _integer(value, name)
    if value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")
    return value


def _metadata_tuple(metadata):
    if metadata in (None, ()):
        return ()
    if isinstance(metadata, Mapping):
        return tuple(metadata.items())
    items = tuple(metadata)
    dict(items)
    return items


def _turn_values(turn: object) -> StateKey:
    if isinstance(turn, Turn):
        return turn.black, turn.white, turn.kings, turn.black_to_move
    if type(turn) is tuple and len(turn) == 4:
        return validate_state_values(*turn)
    return as_turn(turn).as_tuple()


def _finish_turn(turn):
    return Turn(turn.board, 1 - turn.black_to_move, turn.metadata)


_DEFAULT_AMERICAN_RULESET = None
