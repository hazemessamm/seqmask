from __future__ import annotations

import typing

import numpy as np

from seqmask import backend
from seqmask.utils import ensure_non_negative


class HybridSpanMasking:
    """Hybrid scattered + span MLM corruption with Beta-sampled rates.

    Each call:
      1. Sample `mlm_prob` from Beta(mlm_prob * conc, (1 - mlm_prob) * conc).
      2. Sample `alpha` (span share) from Beta similarly, unless
         `span_fraction_mean` is exactly 0 or 1 (degenerate).
      3. `target = round(mlm_prob * eligible_count)`,
         `span_budget = round(alpha * target)`,
         `scatter_budget = target - span_budget`.
      4. Fill span_budget by repeatedly drawing a start position and a
         span length ~ Geometric(span_length_p) capped at max_span_length.
         Spans are clipped at special-token boundaries; specials are
         never included even mid-span.
      5. Fill the remaining gap to `target` with scattered positions from
         the still-eligible (non-special, non-selected) tokens. This
         absorbs both span shortfall and any explicit scatter budget.
      6. Apply 80/10/10 (mask / random / keep) per selected position.

    Args:
        mlm_prob: Mean of the Beta over per-call masking rates. In (0, 1).
        mlm_concentration: Concentration of the rate Beta. > 0.
        span_fraction_mean: Mean of the Beta over alpha (span share).
            In [0, 1]; values 0 / 1 disable the Beta draw.
        span_fraction_concentration: Concentration of the alpha Beta. > 0.
        span_length_p: Geometric distribution parameter; mean span = 1/p.
            In (0, 1].
        max_span_length: Hard cap on individual span length. >= 1.
        masking_prob: Fraction of selected positions replaced with mask.
        mutation_prob: Fraction replaced with a random valid token. The
            remaining `1 - masking_prob - mutation_prob` are left as-is.
        mask_token_id: Token id used to replace masked positions.
        valid_token_ids: Pool of random substitution targets.
        special_token_ids: Tokens that must never be masked.
    """

    def __init__(
        self,
        mlm_prob: float,
        mlm_concentration: float,
        span_fraction_mean: float,
        span_fraction_concentration: float,
        span_length_p: float,
        max_span_length: int,
        masking_prob: float,
        mutation_prob: float,
        mask_token_id: int,
        valid_token_ids: typing.Sequence[int],
        special_token_ids: typing.Sequence[int],
    ):
        if not (0.0 < mlm_prob < 1.0):
            raise ValueError(f"mlm_prob must be in (0, 1), got {mlm_prob}")
        if mlm_concentration <= 0.0:
            raise ValueError(
                f"mlm_concentration must be > 0, got {mlm_concentration}"
            )
        if not (0.0 <= span_fraction_mean <= 1.0):
            raise ValueError(
                f"span_fraction_mean must be in [0, 1], got {span_fraction_mean}"
            )
        if span_fraction_concentration <= 0.0:
            raise ValueError(
                f"span_fraction_concentration must be > 0, got "
                f"{span_fraction_concentration}"
            )
        if not (0.0 < span_length_p <= 1.0):
            raise ValueError(
                f"span_length_p must be in (0, 1], got {span_length_p}"
            )
        if max_span_length < 1:
            raise ValueError(
                f"max_span_length must be >= 1, got {max_span_length}"
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

        self.masking_prob = float(masking_prob)
        self.mutation_prob = float(mutation_prob)
        self.mask_token_id = int(mask_token_id)
        self.span_length_p = float(span_length_p)
        self.max_span_length = int(max_span_length)

        self.valid_token_ids = np.asarray(valid_token_ids, dtype=np.int64)
        self.special_token_ids = np.asarray(special_token_ids, dtype=np.int64)

        self._mlm_beta_alpha = float(mlm_prob * mlm_concentration)
        self._mlm_beta_beta = float((1.0 - mlm_prob) * mlm_concentration)

        self._span_fraction_mean = float(span_fraction_mean)
        self._span_beta_alpha = float(
            span_fraction_mean * span_fraction_concentration
        )
        self._span_beta_beta = float(
            (1.0 - span_fraction_mean) * span_fraction_concentration
        )

    def _fill_spans(
        self,
        row_excluded: np.ndarray,
        span_budget: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample contiguous spans into a boolean row mask.

        Spans are clipped at the first excluded position they encounter so
        specials are never selected. Stops once `span_budget` positions are
        covered or the attempt cap is hit.
        """
        seq_len = row_excluded.shape[0]
        row_selected = np.zeros(seq_len, dtype=bool)
        if span_budget <= 0:
            return row_selected

        covered = 0
        attempts = 0
        max_attempts = max(3 * span_budget, 10)

        while covered < span_budget and attempts < max_attempts:
            attempts += 1
            start = int(rng.integers(0, seq_len))
            if row_excluded[start]:
                continue
            length = min(
                int(rng.geometric(self.span_length_p)),
                self.max_span_length,
            )
            end = min(start + length, seq_len)
            for i in range(start, end):
                if row_excluded[i]:
                    end = i
                    break
            if end <= start:
                continue
            newly_added = int((~row_selected[start:end]).sum())
            row_selected[start:end] = True
            covered += newly_added

        return row_selected

    def __call__(
        self,
        sequences: backend.ArrayLike,
        seed: int | None = None,
        excluded_ids: typing.Iterable[int] | None = None,
    ) -> typing.Tuple[backend.ArrayLike, backend.ArrayLike]:
        """Apply hybrid scatter+span MLM corruption to a batch.

        Args:
            sequences: Array of shape (B, L) of token ids; NumPy or CPU torch.
            seed: If provided, makes the entire call bit-exact reproducible.
            excluded_ids: Token ids that must never be masked. Defaults to
                `self.special_token_ids`.

        Returns:
            `(corrupted_sequences, labels)` with the same shape and type
            as `sequences`. `labels` equals the original token at selected
            positions and -100 elsewhere.
        """
        input_is_not_np = not isinstance(sequences, np.ndarray)
        if input_is_not_np:
            sequences = backend.to_numpy(sequences)

        rng = np.random.default_rng(int(seed) if seed is not None else None)

        mlm_prob = float(rng.beta(self._mlm_beta_alpha, self._mlm_beta_beta))
        if self._span_fraction_mean == 0.0:
            alpha = 0.0
        elif self._span_fraction_mean == 1.0:
            alpha = 1.0
        else:
            alpha = float(
                rng.beta(self._span_beta_alpha, self._span_beta_beta)
            )

        if excluded_ids is None:
            excluded_ids = self.special_token_ids

        sequences = sequences.copy()
        labels = sequences.copy()

        is_excluded = np.zeros_like(sequences, dtype=bool)
        for _id in excluded_ids:
            is_excluded |= sequences == _id

        selected = np.zeros_like(sequences, dtype=bool)
        for b in range(sequences.shape[0]):
            row_excluded = is_excluded[b]
            eligible_count = int((~row_excluded).sum())
            if eligible_count == 0:
                continue
            target_total = int(round(mlm_prob * eligible_count))
            if target_total == 0:
                continue
            span_budget = int(round(alpha * target_total))

            row_selected = self._fill_spans(row_excluded, span_budget, rng)

            shortfall = target_total - int(row_selected.sum())
            if shortfall > 0:
                remaining_eligible = ~row_excluded & ~row_selected
                eligible_indices = np.flatnonzero(remaining_eligible)
                n_scatter = min(shortfall, eligible_indices.size)
                if n_scatter > 0:
                    chosen = rng.choice(
                        eligible_indices, size=n_scatter, replace=False
                    )
                    row_selected[chosen] = True

            selected[b] = row_selected

        labels[~selected] = -100

        random_roll = rng.random(sequences.shape)
        indices_to_mask = selected & (random_roll < self.masking_prob)
        sequences[indices_to_mask] = self.mask_token_id

        mut_hi = self.masking_prob + self.mutation_prob
        indices_to_mutate = (
            selected
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
            sequences = backend.from_numpy(sequences)
            labels = backend.from_numpy(labels)
        return sequences, labels
