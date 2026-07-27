"""Notebook-friendly visualization helpers for native 32-bit boards and rules."""

from typing import Any

from .encoding import MASK32, is_playable_square, square_mask32
from .ruleset import Rule, Ruleset, Turn, as_board, as_turn


def show_board(
    board: object,
    size: float = 3,
    title: str | None = None,
    show: bool = True,
) -> tuple[Any, Any]:
    """Render a board or turn."""

    import matplotlib.pyplot as plt

    if isinstance(board, Turn):
        turn = board
        board = turn.board
        title = title or f"{turn.side} to move"
    else:
        board = as_board(board)

    fig, ax = plt.subplots(1, 1, figsize=(size, size))
    draw_board(ax, board, title=title)
    if show:
        plt.show()
    return fig, ax


def show_turn(
    turn: object,
    size: float = 3,
    title: str | None = None,
    show: bool = True,
) -> tuple[Any, Any]:
    """Render a turn."""

    return show_board(as_turn(turn), size=size, title=title, show=show)


def show_ruleset_rows(
    ruleset: object,
    rules: object | None = None,
    size: float = 3.2,
    title: str | None = None,
    show: bool = True,
) -> list[tuple[Any, Any]]:
    """Render selected rules as separate condition/effect/summary rows."""

    rows = _rule_records(ruleset, rules)
    if not rows:
        raise ValueError("at least one rule row is required")

    figures = []
    for row in rows:
        fig, axes = _show_rule_row(row, size=size, title=_rule_row_title(row, title))
        figures.append((fig, axes))
        if show:
            _display_figure(fig)
            _close_figure(fig)
    return figures


def draw_board(
    ax: Any,
    board: object,
    title: str | None = None,
    dots: object = 0,
) -> Any:
    """Draw a 32-bit board on an existing matplotlib axis."""

    board = as_board(board)
    _draw_base_board(ax)
    _draw_mask_markers(ax, dots, "empty")
    _draw_mask_markers(ax, board.white, "white")
    _draw_mask_markers(ax, board.black, "black")
    _draw_mask_markers(ax, board.kings, "king_overlay")
    if title:
        ax.set_title(title)
    return ax


def _show_rule_row(rule, size=3.2, title=None):
    import matplotlib.pyplot as plt

    columns = ["conditions", "effects", "summary"]
    fig, axes = plt.subplots(
        1,
        len(columns),
        figsize=(size * 3.5, size),
        gridspec_kw={"width_ratios": [1, 1, 0.9]},
        squeeze=True,
    )
    if title:
        fig.suptitle(title, y=0.985, fontsize=9)

    for ax, column in zip(axes, columns, strict=True):
        ax.set_title(column)

    _draw_conditions_board(axes[0], rule)
    _draw_effects_board(axes[1], rule)
    _draw_rule_info(axes[2], rule)

    fig.subplots_adjust(
        left=0.02,
        right=0.99,
        bottom=0.08,
        top=0.76,
        wspace=0.08,
    )
    return fig, axes


def _draw_conditions_board(ax, rule):
    _draw_base_board(ax)
    _draw_mask_markers(ax, rule["is_empty"], "empty")
    _draw_mask_markers(ax, rule["is_white"], "white")
    _draw_mask_markers(ax, rule["is_black"], "black")
    if rule["king"]:
        _draw_mask_markers(ax, _mover_condition_mask(rule), "king_overlay")


def _draw_effects_board(ax, rule):
    _draw_base_board(ax)
    _draw_mask_markers(ax, int(rule["is_black"]) | int(rule["is_white"]), "empty")
    _draw_mask_markers(ax, rule["is_empty"], _side_label(rule))
    if rule["king"] or rule["promotion"]:
        _draw_mask_markers(ax, rule["is_empty"], "king_overlay")


def _draw_rule_info(ax, rule):
    ax.axis("off")
    lines = [
        f"side: {_side_label(rule)}",
        f"king required: {int(rule['king'])}",
        f"capture: {int(rule['capture'])}",
        f"promotion: {int(rule['promotion'])}",
    ]
    ax.text(
        0,
        0.96,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.6,
    )


def _draw_base_board(ax):
    from matplotlib.patches import Rectangle

    ax.set_xlim(-0.02, 8.02)
    ax.set_ylim(-0.02, 8.02)
    ax.set_aspect("equal")
    ax.axis("off")

    for row in range(8):
        for col in range(8):
            y = 7 - row
            fill = "#dddddd" if is_playable_square(row, col) else "white"
            ax.add_patch(Rectangle((col, y), 1, 1, fc=fill, ec="#333333", lw=0.5))

    ax.add_patch(Rectangle((0, 0), 8, 8, fill=False, ec="#222222", lw=1.1, clip_on=False))


def _draw_mask_markers(ax, mask, kind):
    mask = int(mask) & MASK32
    for row in range(8):
        for col in range(8):
            if is_playable_square(row, col) and mask & square_mask32(row, col):
                _draw_mask_marker(ax, col, 7 - row, kind)


def _draw_mask_marker(ax, col, y, kind):
    from matplotlib.patches import Circle

    if kind == "white":
        ax.add_patch(Circle((col + 0.5, y + 0.5), 0.43, fc="#fafafa", ec="#111111", lw=1.15, zorder=3))
    elif kind == "black":
        ax.add_patch(Circle((col + 0.5, y + 0.5), 0.43, fc="#111111", ec="#111111", lw=1.15, zorder=3))
    elif kind == "king_overlay":
        _draw_king_overlay(ax, col, y)
    else:
        ax.add_patch(Circle((col + 0.5, y + 0.5), 0.35, fill=False, ec="#111111", lw=1.05, ls=":", zorder=3))


def _draw_king_overlay(ax, col, y):
    import matplotlib.patheffects as path_effects
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D

    path = TextPath((0, 0), "K", size=1, prop=FontProperties(family="DejaVu Sans", weight="bold"))
    bbox = path.get_extents()
    scale = 0.48 / bbox.height
    transform = (
        Affine2D()
        .translate(-(bbox.x0 + bbox.width / 2), -(bbox.y0 + bbox.height / 2))
        .scale(scale)
        .translate(col + 0.5, y + 0.5)
        + ax.transData
    )
    patch = PathPatch(path, fc="#f2c84b", ec="none", transform=transform, zorder=4)
    patch.set_path_effects([path_effects.withStroke(linewidth=1.15, foreground="#111111")])
    ax.add_patch(patch)


def _display_figure(fig):
    try:
        from IPython.display import display
    except ImportError:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        display(fig)


def _close_figure(fig):
    import matplotlib.pyplot as plt

    plt.close(fig)


def _rule_records(ruleset, rules):
    if isinstance(ruleset, Ruleset):
        selected = ruleset.select_records(rules)
    else:
        selected = list(rules if rules is not None else ruleset)

    rows = []
    for index, rule in enumerate(selected):
        record = rule.as_dict() if isinstance(rule, Rule) else dict(rule)
        record.setdefault("rule_index", index)
        rows.append(record)
    return rows


def _rule_row_title(rule, title=None):
    rule_title = f"rule {rule['rule_index']}"
    return f"{title}: {rule_title}" if title else rule_title


def _mover_condition_mask(rule):
    return rule["is_black"] if int(rule["black_to_move"]) else rule["is_white"]


def _side_label(rule):
    return "black" if int(rule["black_to_move"]) else "white"
