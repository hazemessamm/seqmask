# seqmask

Composable MLM (masked language modeling) corruption strategies with
curriculum schedulers, diffusion-style timestep schedules, and
reference-driven position focusing. Pure NumPy on the inside; accepts
and returns either NumPy arrays or CPU PyTorch tensors transparently.

Built primarily for protein-sequence pretraining but tokenizer-agnostic:
the caller supplies `mask_token_id`, `valid_token_ids`, and
`special_token_ids`, so the same machinery works for any vocabulary.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20532337.svg)](https://doi.org/10.5281/zenodo.20532337)

## Installation

From source:

```bash
pip install seqmask
```

With dev dependencies (pytest):

```bash
pip install -e "seqmask[dev]"
```

Requires Python 3.9+. Depends on `numpy` and `torch` or `jax` (Whatever you are using).

## Quick start

```python
import os
os.environ["SEQMASK_BACKEND"] = "torch" # or `jax`
import torch
from seqmask import BetaDistributionMasking
from seqmask import backend

seqmask = BetaDistributionMasking(
    mlm_prob=0.15,
    mlm_concentration=10.0,
    masking_prob=0.8,           # 80% of selected positions -> [MASK]
    mutation_prob=0.1,          # 10% -> random valid token
                                # remaining 10% -> kept as-is
    mask_token_id=103,
    valid_token_ids=list(range(10, 200)),
    special_token_ids=[0, 1, 2],
)

x = torch.randint(10, 200, (4, 128))
corrupted, labels = seqmask(x, seed=42)
# `corrupted` is the model input; `labels` is -100 outside selected
# positions and holds the original token id at selected positions.
```

## Masking strategies

All four implement the same `__call__(sequences, seed=None, excluded_ids=None, focus_positions=None)` contract
(`DiffusionMasking` additionally takes `timestep`). Inputs and outputs
match types — NumPy in, NumPy out; CPU torch in, CPU torch out.

### `CategoricalMasking`

Sampler-driven per-call rates. Use this when you want curriculum
schedulers (`LinearScheduler`, `EMAScheduler`, `StagedLinearScheduler`)
to steer the masking rate.

```python
from seqmask import (
    CategoricalMasking, MLMProbabilitySampler, LinearScheduler,
)

scheduler = LinearScheduler(
    initial_weights=[1.0, 0.0, 0.0],
    final_weights=[0.0, 0.0, 1.0],
    total_steps=10_000,
)
sampler = MLMProbabilitySampler(
    mlm_probs=[0.15, 0.20, 0.30],
    masking_probs=[0.8, 0.8, 0.8],
    mutation_probs=[0.1, 0.1, 0.1],
    scheduler=scheduler,
)
seqmask = CategoricalMasking(
    mlm_prob_sampler=sampler,
    mask_token_id=103,
    valid_token_ids=list(range(200)),
    special_token_ids=[0, 1, 2],
)

corrupted, labels = seqmask(x)
sampler.step()   # advance the curriculum
```

### `BetaDistributionMasking`

Samples the per-call masking rate from `Beta(mean, concentration)`.
Simplest "randomize the rate" option — no scheduler needed.

### `HybridSpanMasking`

Mix of scattered token masks and contiguous spans (SpanBERT-style).
Per-call: Beta-sample the total rate and the span share `alpha`; fill
the span budget with `Geometric(span_length_p)`-length spans clipped at
specials, then top up with scattered positions to hit the target rate.

```python
from seqmask import HybridSpanMasking

seqmask = HybridSpanMasking(
    mlm_prob=0.15,
    mlm_concentration=10.0,
    span_fraction_mean=0.5,         # ~half of masks come from spans
    span_fraction_concentration=10.0,
    span_length_p=0.2,              # mean span length = 1/p = 5
    max_span_length=10,
    masking_prob=0.8,
    mutation_prob=0.1,
    mask_token_id=103,
    valid_token_ids=list(range(200)),
    special_token_ids=[0, 1, 2],
)
```

### `DiffusionMasking`

Mask-only corruption (no 80/10/10 split) where the rate is a function of
a per-call `timestep`. Used for discrete-diffusion MLM (D3PM / MaskGIT).
Schedules are classes — they pickle cleanly under spawn-based
multiprocessing (PyTorch DataLoader with `num_workers > 0`).

```python
from seqmask import DiffusionMasking, CosineDiffusionScheduler

seqmask = DiffusionMasking(
    total_timesteps=100,
    mask_token_id=103,
    special_token_ids=[0, 1, 2],
    schedule=CosineDiffusionScheduler(),   # or "linear" / "cosine" / "sqrt"
)

t = 42
corrupted, labels = seqmask(x, timestep=t)
```

Available schedulers: `LinearDiffusionScheduler`,
`CosineDiffusionScheduler`, `SqrtDiffusionScheduler`, or any
`DiffusionScheduler` subclass (`__call__(timestep, total_timesteps) -> float`).
Strings `"linear"` / `"cosine"` / `"sqrt"` are convenience aliases that
resolve to the corresponding scheduler instance.

## Focus positions

Bias masking toward specific positions — useful when fine-tuning on
variants relative to a reference (wildtype proteins, edited code, etc.).
All per-token strategies (`Categorical`, `Beta`, `Diffusion`) accept
`focus_positions` as a `(B, L)` boolean mask per call.

```python
from seqmask import (
    BetaDistributionMasking,
    focus_positions_from_reference,
)
from seqmask import backend

reference = backend.to_tensor([10, 20, 30, 40, 50])
variants  = backend.to_tensor([
    [10, 20, 30, 40, 50],   # unchanged
    [10, 99, 30, 99, 50],   # mutated at positions 1 and 3
])

focus = focus_positions_from_reference(variants, reference)
# tensor([[False, False, False, False, False],
#         [False,  True, False,  True, False]])

seqmask = BetaDistributionMasking(
    mlm_prob=0.1,
    mlm_concentration=10.0,
    masking_prob=0.8,
    mutation_prob=0.1,
    mask_token_id=103,
    valid_token_ids=list(range(200)),
    special_token_ids=[0],
    focus_strategy="multiplicative",   # or "force_include" / "weighted"
    focus_strength=3.0,
)
corrupted, labels = seqmask(variants, focus_positions=focus)
```

The three focus strategies:

| `focus_strategy`   | Semantic |
| ------------------ | -------- |
| `"force_include"`  | Focus positions are unconditionally selected (specials still win). |
| `"multiplicative"` | Focus positions are sampled with `min(1, mlm_prob * focus_strength)`. |
| `"weighted"`       | Preserves the total mask budget; redistributes via weighted sampling. |

`focus_positions_from_reference(sequences, reference)` handles broadcasting:
`(L,)`, `(1, L)`, and `(B, L)` references all work; output type matches
`sequences` (NumPy or torch).

## Schedulers (for curriculum sampling)

Use these with `MLMProbabilitySampler` to advance the masking rate over
training steps. They share a `SharedCounter` so multiprocessing workers
see a consistent step count.

- `LinearScheduler(initial_weights, final_weights, total_steps)`
- `EMAScheduler(initial_weights, final_weights, beta=0.999, multiplier=1.0)`
- `StagedLinearScheduler(num_stages, total_steps)` — sequentially
  introduces each candidate, ramping its weight from 0 to equal share.

Call `sampler.step()` (or `scheduler.step()`) once per training step.

## DataLoader integration

All seqmasks, samplers, and schedulers are picklable, so they can be
attached to a `Dataset` and used with `DataLoader(num_workers > 0)`
under both `fork` and `spawn` start methods. For `DiffusionMasking`,
prefer the `DiffusionScheduler` subclasses (or string aliases) over
lambdas — lambdas aren't picklable under spawn.

## Validation

Every constructor that takes weights or probabilities calls
`ensure_non_negative` from `seqmask.utils`. Negative values raise
`ValueError` at construction time; out-of-range sampler outputs raise
at call time. There's no silent clamping.

## Layout

```
seqmask/
  scheduler/           # base Scheduler + Linear / StagedLinear / EMA
  masking/             # CategoricalMasking, BetaDistributionMasking,
                       # HybridSpanMasking, DiffusionMasking +
                       # DiffusionScheduler hierarchy + focus helper
  sampler.py           # MLMProbabilitySampler
  utils.py             # ensure_non_negative, seed_from_components,
                       # create_default_mlm_weights
```

## Testing

```bash
pip install "seqmask[dev]"
pytest
```

## Citation

If you use this software in your research, please cite it as follows:

```bibtex
@software{alsamkary2026seqmask,
  author       = {Hazem Alsamkary},
  title        = {seqmask: r1.0.1},
  month        = jun,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.1},
  doi          = {10.5281/zenodo.20532337},
  url          = {https://doi.org/10.5281/zenodo.20532337}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
