"""Shared helpers for `focus_positions`-aware position selection.

Three strategies, selected at construction time via `focus_strategy`:

- "force_include": focus positions are unconditionally selected
  (still subject to specials). The simplest semantic; expected mask
  rate increases by the focus-position density.

- "multiplicative": focus positions are selected with probability
  `min(1, mlm_prob * focus_strength)` while non-focus positions retain
  `mlm_prob`. Tunable via `focus_strength`.

- "weighted": the expected mask budget `round(mlm_prob * eligible)` is
  redistributed across positions with weights `focus_strength` for
  focus / `1.0` for non-focus, sampled without replacement per row.
  Preserves the overall expected mask rate.
"""

from __future__ import annotations

import typing

import numpy as np

from seqmask import backend

FOCUS_STRATEGIES = ("force_include", "multiplicative", "weighted")


def focus_positions_from_reference(
    sequences: typing.Any,
    reference: typing.Any,
) -> typing.Any:
    """Build a focus_positions boolean mask by comparing sequences to a reference.

    Returns True at every position where `sequences != reference`, ready
    to pass to a seqmask's `focus_positions=` argument. Domain-agnostic:
    the "reference" can be a wildtype protein, a base sentence, an
    original code snippet, or any other per-position template.
    Shapes follow NumPy broadcasting rules:

    - `sequences` `(L,)`   + `reference` `(L,)`   -> `(L,)`
    - `sequences` `(L,)`   + `reference` `(1, L)` -> `(1, L)`
    - `sequences` `(B, L)` + `reference` `(L,)`   -> `(B, L)` (broadcast)
    - `sequences` `(B, L)` + `reference` `(1, L)` -> `(B, L)` (broadcast)
    - `sequences` `(B, L)` + `reference` `(B, L)` -> `(B, L)` (per-row)

    The reference must be no "larger" than `sequences`: a `(B, L)`
    reference paired with a `(L,)` sequences batch is rejected.

    Args:
        sequences: Token ids of shape `(B, L)` or `(L,)`. NumPy array
            or CPU torch tensor.
        reference: Reference token ids; shape-compatible with `sequences`
            per the rules above. Same library as `sequences` or NumPy.

    Returns:
        Boolean array of the broadcast shape. Returns a torch tensor
        when `sequences` is a torch tensor, otherwise a NumPy array.

    Raises:
        ValueError: if the shapes are not broadcast-compatible or the
            last (sequence-length) dimensions disagree.
    """
    input_is_not_np = not isinstance(sequences, np.ndarray)
    seq_np = (
        backend.to_numpy(sequences)
        if input_is_not_np
        else np.asarray(sequences)
    )
    ref_np = (
        backend.to_numpy(reference)
        if not isinstance(reference, np.ndarray)
        else np.asarray(reference)
    )

    if seq_np.ndim not in (1, 2):
        raise ValueError(
            f"sequences must be 1-D or 2-D, got shape {seq_np.shape}"
        )
    if ref_np.ndim not in (1, 2):
        raise ValueError(
            f"reference must be 1-D or 2-D, got shape {ref_np.shape}"
        )
    if seq_np.shape[-1] != ref_np.shape[-1]:
        raise ValueError(
            f"last (length) dimension must match between sequences "
            f"({seq_np.shape}) and reference ({ref_np.shape})"
        )
    if seq_np.ndim == 2 and ref_np.ndim == 2:
        if ref_np.shape[0] not in (1, seq_np.shape[0]):
            raise ValueError(
                f"reference batch dim must be 1 or match sequences "
                f"({seq_np.shape[0]}), got {ref_np.shape[0]}"
            )
    if seq_np.ndim == 1 and ref_np.ndim == 2:
        if ref_np.shape[0] != 1:
            raise ValueError(
                f"reference batch dim must be 1 when sequences is 1-D, "
                f"got {ref_np.shape}"
            )

    result = seq_np != ref_np
    if input_is_not_np:
        return backend.to_tensor(result)
    return result


def validate_focus_params(focus_strategy: str, focus_strength: float) -> None:
    """Validate constructor-time focus parameters."""
    if focus_strategy not in FOCUS_STRATEGIES:
        raise ValueError(
            f"focus_strategy must be one of {list(FOCUS_STRATEGIES)}, "
            f"got {focus_strategy!r}"
        )
    if focus_strength < 0.0:
        raise ValueError(f"focus_strength must be >= 0, got {focus_strength}")


def coerce_focus_positions(
    focus_positions: typing.Optional[typing.Any],
    expected_shape: tuple,
) -> typing.Optional[np.ndarray]:
    """Normalize the caller's focus_positions to a NumPy bool array.

    Accepts NumPy arrays, CPU torch tensors, or Python sequences. Returns
    None if `focus_positions` is None. Raises if shape disagrees with
    `expected_shape`.
    """
    if focus_positions is None:
        return None
    if not isinstance(focus_positions, np.ndarray):
        focus_positions = backend.to_numpy(focus_positions)
    focus_positions = np.asarray(focus_positions, dtype=bool)
    if focus_positions.shape != expected_shape:
        raise ValueError(
            f"focus_positions shape {focus_positions.shape} does not "
            f"match sequences shape {expected_shape}"
        )
    return focus_positions


def select_positions(
    rng: np.random.Generator,
    mlm_prob: float,
    is_excluded: np.ndarray,
    focus_positions: typing.Optional[np.ndarray],
    focus_strategy: str,
    focus_strength: float,
) -> np.ndarray:
    """Return a `(B, L)` boolean mask of selected positions.

    When `focus_positions` is None, falls back to vanilla Bernoulli
    sampling with rate `mlm_prob` (matching the original per-token
    masking behavior). Otherwise applies the selected `focus_strategy`.
    """
    shape = is_excluded.shape

    if focus_positions is None:
        return (rng.random(shape) < mlm_prob) & ~is_excluded

    if focus_strategy == "force_include":
        selected = (rng.random(shape) < mlm_prob) & ~is_excluded
        selected |= focus_positions & ~is_excluded
        return selected

    if focus_strategy == "multiplicative":
        boosted = min(max(mlm_prob * focus_strength, 0.0), 1.0)
        p_per_position = np.where(focus_positions, boosted, mlm_prob)
        return (rng.random(shape) < p_per_position) & ~is_excluded

    if focus_strategy == "weighted":
        selected = np.zeros(shape, dtype=bool)
        for b in range(shape[0]):
            row_eligible = ~is_excluded[b]
            eligible_count = int(row_eligible.sum())
            if eligible_count == 0:
                continue
            target = int(round(mlm_prob * eligible_count))
            if target == 0:
                continue
            target = min(target, eligible_count)
            weights = np.where(focus_positions[b], focus_strength, 1.0)
            weights = weights * row_eligible  # zero out excluded
            total = weights.sum()
            if total == 0:
                continue
            chosen = rng.choice(
                shape[1],
                size=target,
                p=weights / total,
                replace=False,
            )
            selected[b, chosen] = True
        return selected

    # validate_focus_params should have rejected this earlier.
    raise ValueError(f"unknown focus_strategy: {focus_strategy!r}")
