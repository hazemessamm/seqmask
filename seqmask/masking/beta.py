from __future__ import annotations

import typing

import numpy as np

from seqmask import backend
from seqmask.masking.focus import coerce_focus_positions
from seqmask.masking.focus import select_positions
from seqmask.masking.focus import validate_focus_params
from seqmask.utils import ensure_non_negative


class BetaDistributionMasking:
    """BERT-style 80/10/10 MLM corruption with Beta-sampled mask rates.

    Pure NumPy implementation that runs on CPU. Accepts either NumPy
    arrays or CPU PyTorch tensors, and returns the same type as the
    input.

    The masking rate is sampled per call from
    Beta(mlm_prob * mlm_concentration, (1 - mlm_prob) * mlm_concentration),
    which has mean = mlm_prob and concentration = mlm_concentration.
    Higher concentration tightens the distribution around the mean.

    The caller is responsible for assembling `valid_token_ids` (the
    pool of tokens to draw random substitutions from) and
    `special_token_ids` (the set of tokens to exclude from masking).
    This keeps the corruptor tokenizer-agnostic.

    Args:
        mlm_prob: Mean of the Beta distribution over per-call masking
            rates.
        mlm_concentration: Concentration of the Beta distribution.
            Higher values give a tighter distribution around `mlm_prob`.
        masking_prob: Fraction of selected positions replaced with the
            mask token. Typically 0.8.
        mutation_prob: Fraction of selected positions replaced with a
            random token from `valid_token_ids`. Typically 0.1. The
            remaining `1 - masking_prob - mutation_prob` of selected
            positions are left unchanged.
        mask_token_id: Token id used to replace masked positions.
        valid_token_ids: Sequence of token ids that are valid random
            substitution targets. The caller should exclude special
            tokens from this set.
        special_token_ids: Sequence of token ids that must never be
            masked. The caller should include every "do not corrupt"
            token here (e.g., pad, bos/cls, eos, unknown, mask itself,
            and any tokenizer-specific glyphs).
    """

    def __init__(
        self,
        mlm_prob: float,
        mlm_concentration: float,
        masking_prob: float,
        mutation_prob: float,
        mask_token_id: int,
        valid_token_ids: typing.Sequence[int],
        special_token_ids: typing.Sequence[int],
        focus_strategy: str = "multiplicative",
        focus_strength: float = 2.0,
    ):
        if not (0.0 < mlm_prob < 1.0):
            raise ValueError(f"mlm_prob must be in (0, 1), got {mlm_prob}")
        if mlm_concentration <= 0.0:
            raise ValueError(
                f"mlm_concentration must be > 0, got {mlm_concentration}"
            )
        ensure_non_negative(
            [masking_prob, mutation_prob],
            name="(masking_prob, mutation_prob)",
        )
        if masking_prob + mutation_prob > 1.0:
            raise ValueError(
                f"masking_prob + mutation_prob must be <= 1, got "
                f"{masking_prob + mutation_prob}"
            )
        if valid_token_ids is None or len(valid_token_ids) == 0:
            raise ValueError("valid_token_ids must contain at least one id")
        if special_token_ids is None:
            raise ValueError(
                "special_token_ids must be provided (use [] for none)"
            )
        validate_focus_params(focus_strategy, focus_strength)

        self.masking_prob = float(masking_prob)
        self.mutation_prob = float(mutation_prob)
        self.mask_token_id = int(mask_token_id)

        self.valid_token_ids = np.asarray(valid_token_ids, dtype=np.int64)
        self.special_token_ids = np.asarray(special_token_ids, dtype=np.int64)

        self._beta_alpha = float(mlm_prob * mlm_concentration)
        self._beta_beta = float((1.0 - mlm_prob) * mlm_concentration)

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
            seed: If provided, derives all randomness from this seed,
                giving bit-exact reproducibility. If None, uses NumPy's
                default RNG.
            excluded_ids: Token ids that must never be masked.
                Defaults to `self.special_token_ids`.
            focus_positions: Optional `(B, L)` boolean mask of positions
                to bias the seqmask toward. Combined with the configured
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

        rng = np.random.default_rng(int(seed) if seed is not None else None)
        mlm_prob = float(rng.beta(self._beta_alpha, self._beta_beta))

        if excluded_ids is None:
            excluded_ids = self.special_token_ids

        sequences = sequences.copy()
        labels = sequences.copy()

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

        indices_to_mask = masked_indices & (random_roll < self.masking_prob)
        sequences[indices_to_mask] = self.mask_token_id

        mut_hi = self.masking_prob + self.mutation_prob
        indices_to_mutate = (
            masked_indices
            & (random_roll >= self.masking_prob)
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
