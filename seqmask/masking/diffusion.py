from __future__ import annotations

import math
import typing

import numpy as np

from seqmask import backend
from seqmask.masking.focus import coerce_focus_positions
from seqmask.masking.focus import select_positions
from seqmask.masking.focus import validate_focus_params


class DiffusionScheduler:
    """Base class for timestep → masking-probability mappings.

    Subclasses implement `__call__(timestep, total_timesteps) -> float`,
    returning the masking probability at the requested step. The result
    must lie in `[0, 1]`; `DiffusionMasking` will raise if it does not.

    Instances of subclasses are picklable as long as their `__dict__`
    state is picklable, so a `DiffusionMasking` carrying a scheduler
    instance can be sent to PyTorch DataLoader workers under spawn-based
    multiprocessing without issue.
    """

    def __call__(
        self,
        timestep: float,
        total_timesteps: float,
    ) -> float:
        raise NotImplementedError


class LinearDiffusionScheduler(DiffusionScheduler):
    """p(t) = t / T."""

    def __call__(self, timestep: float, total_timesteps: float) -> float:
        return timestep / total_timesteps


class CosineDiffusionScheduler(DiffusionScheduler):
    """MaskGIT-style cosine: p(t) = 1 - cos((t / T) * pi / 2)."""

    def __call__(self, timestep: float, total_timesteps: float) -> float:
        return 1.0 - math.cos((timestep / total_timesteps) * math.pi / 2.0)


class SqrtDiffusionScheduler(DiffusionScheduler):
    """p(t) = sqrt(t / T)."""

    def __call__(self, timestep: float, total_timesteps: float) -> float:
        return math.sqrt(timestep / total_timesteps)


_BUILTIN_SCHEDULERS: typing.Dict[str, typing.Type[DiffusionScheduler]] = {
    "linear": LinearDiffusionScheduler,
    "cosine": CosineDiffusionScheduler,
    "sqrt": SqrtDiffusionScheduler,
}


class DiffusionMasking:
    """Mask-only diffusion-style corruption with a timestep-driven rate.

    Used in discrete diffusion MLM (D3PM, MaskGIT). The masking
    probability is derived from the per-call `timestep` and the
    constructor-time `total_timesteps` via a `DiffusionScheduler`.
    Every selected position is replaced with `mask_token_id` — no
    80/10/10 split, no random-token mutation. This matches the
    absorbing-state forward process used in mask-token diffusion.

    Endpoints (with the built-in schedulers):
      * t = 0 → p = 0 → no masking.
      * t = T → p = 1 → every eligible position masked.

    Args:
        total_timesteps: Maximum timestep `T`. Must be > 0.
        mask_token_id: Token id used to replace selected positions.
        special_token_ids: Tokens that must never be masked. Pass `[]`
            for no protection.
        schedule: Either a `DiffusionScheduler` instance, a string
            naming a built-in scheduler ("linear", "cosine", "sqrt"),
            or any callable `(timestep, total_timesteps) -> float in
            [0, 1]`. Prefer scheduler subclass instances when using
            PyTorch DataLoader workers — lambdas and locally-defined
            closures are not picklable under spawn-based multiprocessing.
    """

    def __init__(
        self,
        total_timesteps: int,
        mask_token_id: int,
        special_token_ids: typing.Sequence[int],
        schedule: typing.Union[
            DiffusionScheduler,
            str,
            typing.Callable[[float, float], float],
        ] = "cosine",
        focus_strategy: str = "multiplicative",
        focus_strength: float = 2.0,
    ):
        if total_timesteps <= 0:
            raise ValueError(
                f"total_timesteps must be > 0, got {total_timesteps}"
            )
        if special_token_ids is None:
            raise ValueError(
                "special_token_ids must be provided (use [] for none)"
            )
        validate_focus_params(focus_strategy, focus_strength)

        if isinstance(schedule, str):
            if schedule not in _BUILTIN_SCHEDULERS:
                raise ValueError(
                    f"unknown schedule '{schedule}'. "
                    f"Choose one of {sorted(_BUILTIN_SCHEDULERS)} "
                    f"or pass a DiffusionScheduler instance / callable."
                )
            self.schedule = _BUILTIN_SCHEDULERS[schedule]()
        elif callable(schedule):
            self.schedule = schedule
        else:
            raise ValueError(
                "schedule must be a DiffusionScheduler, a string name, "
                f"or a callable; got {type(schedule).__name__}"
            )

        self.total_timesteps = int(total_timesteps)
        self.mask_token_id = int(mask_token_id)
        self.special_token_ids = np.asarray(special_token_ids, dtype=np.int64)
        self.focus_strategy = focus_strategy
        self.focus_strength = float(focus_strength)

    def mlm_prob_at(self, timestep: typing.Union[int, float]) -> float:
        """Return the masking rate prescribed by the schedule at `timestep`.

        Raises if `timestep` is outside `[0, total_timesteps]` or if the
        schedule returns a value outside `[0, 1]`.
        """
        if not (0 <= timestep <= self.total_timesteps):
            raise ValueError(
                f"timestep must be in [0, {self.total_timesteps}], "
                f"got {timestep}"
            )
        p = float(self.schedule(float(timestep), float(self.total_timesteps)))
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"schedule returned mlm_prob outside [0, 1]: {p}")
        return p

    def __call__(
        self,
        sequences: backend.ArrayLike,
        timestep: typing.Union[int, float],
        seed: int | None = None,
        excluded_ids: typing.Iterable[int] | None = None,
        focus_positions: typing.Optional[backend.ArrayLike] = None,
    ) -> typing.Tuple[backend.ArrayLike, backend.ArrayLike]:
        """Apply diffusion-style masking at the given timestep.

        Args:
            sequences: Array of shape (B, L) of token ids; NumPy or CPU torch.
            timestep: Current diffusion timestep in `[0, total_timesteps]`.
            seed: If provided, makes the call bit-exact reproducible.
            excluded_ids: Tokens that must never be masked. Defaults to
                `self.special_token_ids`.
            focus_positions: Optional `(B, L)` boolean mask of positions
                to bias the seqmask toward. Combined with the configured
                `focus_strategy` / `focus_strength`. None disables
                focusing for this call. Note: under "force_include" the
                seqmask will mask focus positions even at `timestep=0`,
                which technically diverges from the strict diffusion
                forward-process semantics.

        Returns:
            `(corrupted_sequences, labels)` with the same shape and type
            as `sequences`. `labels` equals the original token at masked
            positions and -100 elsewhere.
        """
        input_is_not_np = not isinstance(sequences, np.ndarray)
        if input_is_not_np:
            sequences = backend.to_numpy(sequences)

        focus_positions = coerce_focus_positions(
            focus_positions, sequences.shape
        )

        mlm_prob = self.mlm_prob_at(timestep)
        rng = np.random.default_rng(int(seed) if seed is not None else None)

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

        print(masked_indices.shape)

        labels[~masked_indices] = -100
        sequences[masked_indices] = self.mask_token_id

        if input_is_not_np:
            sequences = backend.to_tensor(sequences)
            labels = backend.to_tensor(labels)
        return sequences, labels
