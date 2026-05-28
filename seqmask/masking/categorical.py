from __future__ import annotations

import typing

import numpy as np

from seqmask import backend
from seqmask.masking.focus import coerce_focus_positions
from seqmask.masking.focus import select_positions
from seqmask.masking.focus import validate_focus_params
from seqmask.sampler import MLMProbabilitySampler
from seqmask.utils import ensure_non_negative


class CategoricalMasking:
    """BERT-style 80/10/10 MLM corruption with sampler-driven per-call rates.

    Pure NumPy implementation that runs on CPU. Accepts either NumPy
    arrays or CPU PyTorch tensors, and returns the same type as the
    input.

    Per-call probabilities (mlm_prob, masking_prob, mutation_prob) are
    drawn from `mlm_probability_sampler`, which must implement a
    `sample(seed: int | None)` method returning a 3-tuple of floats.
    This makes the class agnostic to the specific sampling strategy
    (curriculum schedules, fixed grids, Beta distributions, etc.).

    The caller is responsible for assembling `valid_token_ids` (the
    pool of tokens to draw random substitutions from) and
    `special_token_ids` (the set of tokens to exclude from masking).
    This keeps the class tokenizer-agnostic.

    Args:
        mlm_probability_sampler: Object with a `sample(seed)` method
            returning `(mlm_prob, masking_prob, mutation_prob)` per call.
        mask_token_id: Token id used to replace masked positions.
        valid_token_ids: Sequence of token ids that are valid random
            substitution targets. The caller should exclude special
            tokens from this set.
        special_token_ids: Sequence of token ids that must never be
            masked. The caller should include every "do not corrupt"
            token here.
    """

    def __init__(
        self,
        mlm_prob_sampler: MLMProbabilitySampler,
        mask_token_id: int,
        valid_token_ids: typing.Sequence[int],
        special_token_ids: typing.Sequence[int],
        focus_strategy: str = "multiplicative",
        focus_strength: float = 2.0,
    ):
        if mlm_prob_sampler is None:
            raise ValueError("mlm_prob_sampler is required")
        if not hasattr(mlm_prob_sampler, "sample"):
            raise ValueError(
                "mlm_probability_sampler must have a 'sample(seed)' method"
            )
        if valid_token_ids is None or len(valid_token_ids) == 0:
            raise ValueError("valid_token_ids must contain at least one id")
        if special_token_ids is None:
            raise ValueError(
                "special_token_ids must be provided (use [] for none)"
            )
        validate_focus_params(focus_strategy, focus_strength)

        self.mlm_prob_sampler = mlm_prob_sampler
        self.mask_token_id = int(mask_token_id)
        self.valid_token_ids = np.asarray(valid_token_ids, dtype=np.int64)
        self.special_token_ids = np.asarray(special_token_ids, dtype=np.int64)
        self.focus_strategy = focus_strategy
        self.focus_strength = float(focus_strength)

    def __call__(
        self,
        sequences: backend.ArrayLike,
        seed: int | None = None,
        excluded_ids: typing.Iterable[int] | None = None,
        focus_positions: typing.Optional[backend.ArrayLike] = None,
    ) -> typing.Tuple[backend.ArrayLike, backend.ArrayLike]:
        """Apply MLM corruption to a batch of token sequences.

        Args:
            sequences: Array of shape (B, L) of token ids. Can be a
                NumPy ndarray or a CPU PyTorch tensor.
            seed: If provided, derives all randomness (sampler + position
                draws) from this seed, giving bit-exact reproducibility.
                If None, uses fresh RNGs.
            excluded_ids: Token ids that must never be masked.
                Defaults to `self.special_token_ids`.
            focus_positions: Optional `(B, L)` boolean mask of positions
                to bias the seqmask toward (e.g., positions that differ
                from a reference sequence). Combined with the configured
                `focus_strategy` / `focus_strength`. None disables
                focusing for this call.

        Returns:
            `(corrupted_sequences, labels)`. Both have shape (B, L) and
            the same type as the input. `labels` equals the original
            token at MLM-selected positions and -100 elsewhere.
        """
        input_is_not_np = not isinstance(sequences, np.ndarray)
        if input_is_not_np:
            sequences = backend.to_numpy(sequences)

        focus_positions = coerce_focus_positions(
            focus_positions, sequences.shape
        )

        mlm_prob, masking_prob, mutation_prob = self.mlm_prob_sampler.sample(
            seed=seed
        )

        if not (0.0 <= mlm_prob <= 1.0):
            raise ValueError(
                f"sampler returned mlm_prob outside [0, 1]: {mlm_prob}"
            )
        ensure_non_negative(
            [masking_prob, mutation_prob],
            name="sampler (masking_prob, mutation_prob)",
        )
        if masking_prob + mutation_prob > 1.0:
            raise ValueError(
                f"sampler returned masking_prob + mutation_prob > 1: "
                f"{masking_prob} + {mutation_prob}"
            )

        rng = np.random.default_rng(int(seed) if seed is not None else None)

        if excluded_ids is None:
            excluded_ids = self.special_token_ids
        sequences, labels = sequences.copy(), sequences.copy()

        is_excluded = np.zeros_like(sequences, dtype=bool)
        for _id in excluded_ids:
            is_excluded |= sequences == _id

        masked_indices = select_positions(
            rng=rng,
            mlm_prob=mlm_prob,
            is_excluded=is_excluded,
            focus_positions=focus_positions,
            focus_strategy=self.focus_strategy,
            focus_strength=self.focus_strength,
        )

        labels[~masked_indices] = -100

        random_roll = rng.random(sequences.shape)

        indices_to_mask = masked_indices & (random_roll < masking_prob)
        sequences[indices_to_mask] = self.mask_token_id

        mut_hi = masking_prob + mutation_prob
        indices_to_mutate = (
            masked_indices
            & (random_roll >= masking_prob)
            & (random_roll < mut_hi)
        )

        if indices_to_mutate.any():
            random_indices = rng.integers(
                0, len(self.valid_token_ids), sequences.shape
            )
            sequences[indices_to_mutate] = self.valid_token_ids[
                random_indices[indices_to_mutate]
            ]

        if input_is_not_np:
            sequences = backend.to_tensor(sequences)
            labels = backend.to_tensor(labels)
        return sequences, labels
