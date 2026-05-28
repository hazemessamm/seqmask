import os

os.environ[
    "SEQMASK_BACKEND"
] = "torch"  # force PyTorch for deterministic testing


import numpy as np
import pytest

from seqmask.sampler import MLMProbabilitySampler
from seqmask.scheduler import LinearScheduler
from seqmask.scheduler import Scheduler


class _ConstantWeightScheduler(Scheduler):
    """Test double that returns a fixed weight vector each call."""

    def __init__(self, weights):
        super().__init__(initial_value=0, final_value=1)
        self._weights = list(weights)

    def sample(self):
        return list(self._weights)


class TestMLMProbabilitySamplerInit:
    def test_accepts_equal_length_lists(self):
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1, 0.2],
            masking_probs=[0.8, 0.8],
            mutation_probs=[0.1, 0.1],
        )
        assert sampler.num_candidates == 2

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            MLMProbabilitySampler(
                mlm_probs=[0.1, 0.2],
                masking_probs=[0.8],
                mutation_probs=[0.1, 0.1],
            )

    def test_rejects_empty_lists(self):
        with pytest.raises(ValueError):
            MLMProbabilitySampler(
                mlm_probs=[], masking_probs=[], mutation_probs=[]
            )

    @pytest.mark.parametrize(
        "mlm,mask,mut",
        [
            ([-0.1, 0.2], [0.8, 0.8], [0.1, 0.1]),
            ([0.1, 0.2], [-0.1, 0.8], [0.1, 0.1]),
            ([0.1, 0.2], [0.8, 0.8], [0.1, -0.05]),
        ],
    )
    def test_rejects_negative_probabilities(self, mlm, mask, mut):
        with pytest.raises(ValueError):
            MLMProbabilitySampler(
                mlm_probs=mlm, masking_probs=mask, mutation_probs=mut
            )

    def test_stores_arrays_as_numpy(self):
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1], masking_probs=[0.8], mutation_probs=[0.1]
        )
        assert isinstance(sampler.mlm_probs, np.ndarray)
        assert isinstance(sampler.masking_probs, np.ndarray)
        assert isinstance(sampler.mutation_probs, np.ndarray)


class TestMLMProbabilitySamplerSampleByIndex:
    def test_returns_triple_at_index(self):
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1, 0.2, 0.3],
            masking_probs=[0.7, 0.8, 0.9],
            mutation_probs=[0.05, 0.1, 0.05],
        )
        assert sampler.sample(index=1) == (
            pytest.approx(0.2),
            pytest.approx(0.8),
            pytest.approx(0.1),
        )

    def test_negative_index_rejected(self):
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1, 0.2],
            masking_probs=[0.8, 0.8],
            mutation_probs=[0.1, 0.1],
        )
        with pytest.raises(ValueError):
            sampler.sample(index=-1)

    def test_out_of_range_index_rejected(self):
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1, 0.2],
            masking_probs=[0.8, 0.8],
            mutation_probs=[0.1, 0.1],
        )
        with pytest.raises(ValueError):
            sampler.sample(index=2)

    def test_index_returns_floats(self):
        sampler = MLMProbabilitySampler(
            mlm_probs=np.array([0.1]),
            masking_probs=np.array([0.8]),
            mutation_probs=np.array([0.1]),
        )
        mlm, mask, mut = sampler.sample(index=0)
        assert isinstance(mlm, float)
        assert isinstance(mask, float)
        assert isinstance(mut, float)


class TestMLMProbabilitySamplerSampleRandom:
    def test_returns_one_of_candidates(self):
        mlm = [0.1, 0.2, 0.3]
        sampler = MLMProbabilitySampler(
            mlm_probs=mlm,
            masking_probs=[0.8] * 3,
            mutation_probs=[0.1] * 3,
        )
        for _ in range(20):
            value, _, _ = sampler.sample()
            assert value in mlm

    def test_seed_makes_random_sampling_reproducible(self):
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1, 0.2, 0.3, 0.4],
            masking_probs=[0.8] * 4,
            mutation_probs=[0.1] * 4,
        )
        a = sampler.sample(seed=123)
        b = sampler.sample(seed=123)
        assert a == b

    def test_different_seeds_can_differ(self):
        sampler = MLMProbabilitySampler(
            mlm_probs=list(np.linspace(0.05, 0.5, 10)),
            masking_probs=[0.8] * 10,
            mutation_probs=[0.1] * 10,
        )
        results = {sampler.sample(seed=s) for s in range(20)}
        # We expect more than one distinct triple across 20 different seeds.
        assert len(results) > 1


class TestMLMProbabilitySamplerSampleFromScheduler:
    def test_picks_index_with_dominant_weight(self):
        # Weights all on the last index → must pick last triple.
        sched = _ConstantWeightScheduler([0.0, 0.0, 1.0])
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1, 0.2, 0.3],
            masking_probs=[0.8, 0.8, 0.8],
            mutation_probs=[0.1, 0.1, 0.1],
            scheduler=sched,
        )
        mlm, _, _ = sampler.sample(seed=0)
        assert mlm == pytest.approx(0.3)

    def test_normalizes_unnormalized_weights(self):
        # Weights sum to 4, not 1; still must work.
        sched = _ConstantWeightScheduler([0.0, 4.0, 0.0])
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1, 0.2, 0.3],
            masking_probs=[0.8] * 3,
            mutation_probs=[0.1] * 3,
            scheduler=sched,
        )
        for seed in range(5):
            mlm, _, _ = sampler.sample(seed=seed)
            assert mlm == pytest.approx(0.2)

    def test_rejects_negative_weights_from_scheduler(self):
        # Any negative weight returned by the scheduler at runtime must
        # raise instead of being silently clamped.
        sched = _ConstantWeightScheduler([-1.0, 1.0, -2.0])
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1, 0.2, 0.3],
            masking_probs=[0.8] * 3,
            mutation_probs=[0.1] * 3,
            scheduler=sched,
        )
        with pytest.raises(ValueError):
            sampler.sample(seed=0)

    def test_all_zero_weights_falls_back_to_uniform(self):
        sched = _ConstantWeightScheduler([0.0, 0.0, 0.0])
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1, 0.2, 0.3],
            masking_probs=[0.8] * 3,
            mutation_probs=[0.1] * 3,
            scheduler=sched,
        )
        seen = set()
        for seed in range(50):
            mlm, _, _ = sampler.sample(seed=seed)
            seen.add(mlm)
        # Uniform fallback should explore all candidates over 50 seeds.
        assert seen == {0.1, 0.2, 0.3}

    def test_seed_makes_scheduler_sampling_reproducible(self):
        sched = _ConstantWeightScheduler([0.5, 0.5, 0.0])
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1, 0.2, 0.3],
            masking_probs=[0.8] * 3,
            mutation_probs=[0.1] * 3,
            scheduler=sched,
        )
        assert sampler.sample(seed=42) == sampler.sample(seed=42)

    def test_explicit_index_overrides_scheduler(self):
        # Scheduler would pick index 0, but index=2 was passed explicitly.
        sched = _ConstantWeightScheduler([1.0, 0.0, 0.0])
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1, 0.2, 0.3],
            masking_probs=[0.8] * 3,
            mutation_probs=[0.1] * 3,
            scheduler=sched,
        )
        mlm, _, _ = sampler.sample(index=2)
        assert mlm == pytest.approx(0.3)


class TestMLMProbabilitySamplerStepReset:
    def test_step_without_scheduler_is_noop(self):
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1], masking_probs=[0.8], mutation_probs=[0.1]
        )
        # Should not raise.
        sampler.step()
        sampler.reset()

    def test_step_delegates_to_scheduler(self):
        sched = LinearScheduler(
            initial_weights=[0.0],
            final_weights=[1.0],
            total_steps=10,
        )
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1],
            masking_probs=[0.8],
            mutation_probs=[0.1],
            scheduler=sched,
        )
        sampler.step()
        sampler.step()
        assert sched.current_step.value == 2

    def test_reset_delegates_to_scheduler(self):
        sched = LinearScheduler(
            initial_weights=[0.0],
            final_weights=[1.0],
            total_steps=10,
        )
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1],
            masking_probs=[0.8],
            mutation_probs=[0.1],
            scheduler=sched,
        )
        sampler.step()
        sampler.step()
        sampler.reset()
        assert sched.current_step.value == 0


class TestSampleFromSchedulerDirect:
    def test_sample_from_scheduler_is_int(self):
        sched = _ConstantWeightScheduler([0.0, 1.0, 0.0])
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1, 0.2, 0.3],
            masking_probs=[0.8] * 3,
            mutation_probs=[0.1] * 3,
            scheduler=sched,
        )
        idx = sampler.sample_from_scheduler(seed=0)
        assert isinstance(idx, int)
        assert idx == 1

    def test_sample_randomly_is_int_and_in_range(self):
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.1, 0.2, 0.3],
            masking_probs=[0.8] * 3,
            mutation_probs=[0.1] * 3,
        )
        for s in range(20):
            idx = sampler.sample_randomly(seed=s)
            assert isinstance(idx, int)
            assert 0 <= idx < sampler.num_candidates
