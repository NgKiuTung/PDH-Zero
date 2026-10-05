"""Pure Mini-PDH v0 hand validation and pattern scoring."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from .types import HONOR_TILE_TYPES, TERMINAL_HONOR_TYPES, TILE_TYPES, WinResult


class RuleError(ValueError):
    """Raised when a hand violates a Mini-PDH structural invariant."""


def validate_counts(counts: Iterable[int]) -> tuple[int, ...]:
    """Validate and normalize one 13-type tile-count vector."""
    normalized = tuple(int(value) for value in counts)
    if len(normalized) != TILE_TYPES:
        raise RuleError(f"Expected {TILE_TYPES} tile counts, got {len(normalized)}")
    if any(value < 0 or value > 4 for value in normalized):
        raise RuleError(f"Tile counts must be in [0, 4]: {normalized}")
    return normalized


@lru_cache(maxsize=100_000)
def _meld_partitions(counts: tuple[int, ...], melds_needed: int) -> tuple[tuple[str, ...], ...]:
    """Enumerate triplet/sequence decompositions for a fixed remaining meld count."""
    if melds_needed == 0:
        return ((),) if sum(counts) == 0 else ()
    if sum(counts) != melds_needed * 3:
        return ()
    first = next((index for index, value in enumerate(counts) if value), None)
    if first is None:
        return ()
    partitions: list[tuple[str, ...]] = []
    mutable = list(counts)
    if mutable[first] >= 3:
        mutable[first] -= 3
        for suffix in _meld_partitions(tuple(mutable), melds_needed - 1):
            partitions.append(("TRIPLET",) + suffix)
        mutable[first] += 3
    if first <= 6 and mutable[first + 1] > 0 and mutable[first + 2] > 0:
        mutable[first] -= 1
        mutable[first + 1] -= 1
        mutable[first + 2] -= 1
        for suffix in _meld_partitions(tuple(mutable), melds_needed - 1):
            partitions.append(("SEQUENCE",) + suffix)
    return tuple(partitions)


def _standard_decompositions(counts: tuple[int, ...], melds_needed: int) -> tuple[tuple[str, ...], ...]:
    """Enumerate valid pair-plus-meld structures for concealed tiles."""
    expected = 2 + 3 * melds_needed
    if sum(counts) != expected:
        return ()
    decompositions: list[tuple[str, ...]] = []
    for pair_tile, amount in enumerate(counts):
        if amount < 2:
            continue
        mutable = list(counts)
        mutable[pair_tile] -= 2
        decompositions.extend(_meld_partitions(tuple(mutable), melds_needed))
    return tuple(decompositions)


def _is_four_pairs(counts: tuple[int, ...], open_pongs: tuple[int, ...]) -> bool:
    """Return whether four pairs are present, counting four identical tiles as two pairs."""
    return not open_pongs and sum(counts) == 8 and all(value % 2 == 0 for value in counts) and sum(
        value // 2 for value in counts
    ) == 4


def evaluate_win(
    concealed_counts: Iterable[int],
    open_pongs: Iterable[int],
    pattern_multipliers: dict[str, int],
) -> WinResult | None:
    """Return the highest-scoring legal Mini-PDH win, or ``None`` when invalid."""
    counts = validate_counts(concealed_counts)
    pongs = tuple(int(tile) for tile in open_pongs)
    if any(tile < 0 or tile >= TILE_TYPES for tile in pongs):
        raise RuleError(f"Invalid open-pong tile: {pongs}")
    if len(pongs) > 2:
        return None
    decompositions = _standard_decompositions(counts, 2 - len(pongs))
    four_pairs = _is_four_pairs(counts, pongs)
    if not decompositions and not four_pairs:
        return None

    present_tiles = {index for index, value in enumerate(counts) if value > 0}
    present_tiles.update(pongs)
    patterns: set[str] = set()
    if decompositions and any(all(kind == "TRIPLET" for kind in item) for item in decompositions):
        patterns.add("PONG_PONG")
    if not pongs:
        patterns.add("CLOSED")
    if four_pairs:
        patterns.add("FOUR_PAIRS")
    if present_tiles and all(tile < 9 for tile in present_tiles):
        patterns.add("PURE_NUMBERS")
    if present_tiles and all(tile in HONOR_TILE_TYPES for tile in present_tiles):
        patterns.add("ALL_HONORS")
    if present_tiles and all(tile in TERMINAL_HONOR_TYPES for tile in present_tiles):
        patterns.add("TERMINALS_HONORS")
    scored = [(pattern_multipliers[name], name) for name in patterns if name in pattern_multipliers]
    if not scored:
        return None
    multiplier, selected = max(scored, key=lambda item: (item[0], item[1]))
    ordered_patterns = tuple(sorted(name for _, name in scored))
    return WinResult(multiplier=multiplier, patterns=ordered_patterns, selected_pattern=selected)
