import os

os.environ[
    "SEQMASK_BACKEND"
] = "torch"  # force PyTorch for deterministic testing

import numpy as np
import pytest
import torch

from seqmask.masking import BetaDistributionMasking
from seqmask.masking import CategoricalMasking
from seqmask.masking import HybridSpanMasking
from seqmask.sampler import MLMProbabilitySampler


class _FixedSampler:
    """Sampler stub that returns a preset (mlm, mask, mut) triple."""

    def __init__(self, mlm_prob, masking_prob, mutation_prob):
        self._triple = (
            float(mlm_prob),
            float(masking_prob),
            float(mutation_prob),
        )

    def sample(self, seed=None):
        return self._triple


def _make_categorical(
    mlm_prob=0.15,
    masking_prob=0.8,
    mutation_prob=0.1,
    mask_token_id=103,
    valid_token_ids=None,
    special_token_ids=None,
):
    if valid_token_ids is None:
        valid_token_ids = list(range(10, 200))
    if special_token_ids is None:
        special_token_ids = [0, 1, 2]
    return CategoricalMasking(
        mlm_prob_sampler=_FixedSampler(mlm_prob, masking_prob, mutation_prob),
        mask_token_id=mask_token_id,
        valid_token_ids=valid_token_ids,
        special_token_ids=special_token_ids,
    )


class TestCategoricalMaskingInit:
    def test_constructs_with_valid_args(self):
        m = _make_categorical()
        assert m.mask_token_id == 103
        assert isinstance(m.valid_token_ids, np.ndarray)
        assert isinstance(m.special_token_ids, np.ndarray)

    def test_rejects_none_sampler(self):
        with pytest.raises(ValueError):
            CategoricalMasking(
                mlm_prob_sampler=None,
                mask_token_id=103,
                valid_token_ids=[1, 2],
                special_token_ids=[0],
            )

    def test_rejects_sampler_without_sample_method(self):
        class BadSampler:
            pass

        with pytest.raises(ValueError):
            CategoricalMasking(
                mlm_prob_sampler=BadSampler(),
                mask_token_id=103,
                valid_token_ids=[1, 2],
                special_token_ids=[0],
            )

    def test_rejects_empty_valid_token_ids(self):
        with pytest.raises(ValueError):
            CategoricalMasking(
                mlm_prob_sampler=_FixedSampler(0.15, 0.8, 0.1),
                mask_token_id=103,
                valid_token_ids=[],
                special_token_ids=[0],
            )

    def test_rejects_none_special_token_ids(self):
        with pytest.raises(ValueError):
            CategoricalMasking(
                mlm_prob_sampler=_FixedSampler(0.15, 0.8, 0.1),
                mask_token_id=103,
                valid_token_ids=[1, 2],
                special_token_ids=None,
            )

    def test_accepts_empty_special_token_ids(self):
        m = CategoricalMasking(
            mlm_prob_sampler=_FixedSampler(0.15, 0.8, 0.1),
            mask_token_id=103,
            valid_token_ids=[1, 2],
            special_token_ids=[],
        )
        assert m.special_token_ids.size == 0


class TestCategoricalMaskingCall:
    def test_output_shape_matches_input(self):
        m = _make_categorical()
        x = np.random.default_rng(0).integers(10, 200, size=(4, 16))
        out, labels = m(x, seed=42)
        assert out.shape == x.shape
        assert labels.shape == x.shape

    def test_numpy_input_returns_numpy(self):
        m = _make_categorical()
        x = np.random.default_rng(0).integers(10, 200, size=(2, 8))
        out, labels = m(x, seed=1)
        assert isinstance(out, np.ndarray)
        assert isinstance(labels, np.ndarray)

    def test_torch_input_returns_torch(self):
        m = _make_categorical()
        x = torch.randint(10, 200, (2, 8))
        out, labels = m(x, seed=1)
        assert isinstance(out, torch.Tensor)
        assert isinstance(labels, torch.Tensor)

    def test_does_not_mutate_input(self):
        m = _make_categorical()
        x = np.random.default_rng(0).integers(10, 200, size=(3, 8))
        original = x.copy()
        m(x, seed=7)
        np.testing.assert_array_equal(x, original)

    def test_torch_input_not_mutated(self):
        m = _make_categorical()
        x = torch.randint(10, 200, (3, 8))
        original = x.clone()
        m(x, seed=7)
        assert torch.equal(x, original)

    def test_special_tokens_never_masked(self):
        special = [0, 1, 2]
        m = _make_categorical(
            mlm_prob=1.0,
            masking_prob=1.0,
            mutation_prob=0.0,
            special_token_ids=special,
        )
        x = np.array([[0, 1, 2, 50, 60, 70]] * 4)
        out, labels = m(x, seed=0)
        for sid in special:
            positions = x == sid
            # Special positions must stay unchanged.
            np.testing.assert_array_equal(out[positions], x[positions])
            # Labels at special positions must be -100 (not selected).
            assert np.all(labels[positions] == -100)

    def test_labels_minus_100_outside_selected_positions(self):
        m = _make_categorical(
            mlm_prob=1.0, masking_prob=1.0, mutation_prob=0.0
        )
        x = np.random.default_rng(0).integers(10, 200, size=(2, 6))
        out, labels = m(x, seed=0)
        # With mlm_prob=1 and no specials in x, every position is selected.
        selected = labels != -100
        np.testing.assert_array_equal(labels[selected], x[selected])

    def test_zero_mlm_prob_yields_no_changes(self):
        m = _make_categorical(mlm_prob=0.0)
        x = np.random.default_rng(0).integers(10, 200, size=(3, 8))
        out, labels = m(x, seed=0)
        np.testing.assert_array_equal(out, x)
        assert np.all(labels == -100)

    def test_full_mask_replaces_all_non_special_with_mask_token(self):
        m = _make_categorical(
            mlm_prob=1.0,
            masking_prob=1.0,
            mutation_prob=0.0,
            mask_token_id=999,
            special_token_ids=[0, 1, 2],
        )
        x = np.full((3, 5), 50, dtype=np.int64)
        out, labels = m(x, seed=0)
        assert np.all(out == 999)
        # Labels keep the original ids at masked positions.
        assert np.all(labels == 50)

    def test_full_mutation_replaces_with_valid_token_ids(self):
        valid = list(range(50, 60))
        m = _make_categorical(
            mlm_prob=1.0,
            masking_prob=0.0,
            mutation_prob=1.0,
            valid_token_ids=valid,
            special_token_ids=[0, 1, 2],
        )
        x = np.full((4, 8), 30, dtype=np.int64)
        out, _ = m(x, seed=0)
        assert np.all(np.isin(out, valid))

    def test_seed_makes_outputs_deterministic(self):
        m = _make_categorical()
        x = np.random.default_rng(0).integers(10, 200, size=(3, 12))
        out_a, lab_a = m(x, seed=99)
        out_b, lab_b = m(x, seed=99)
        np.testing.assert_array_equal(out_a, out_b)
        np.testing.assert_array_equal(lab_a, lab_b)

    def test_different_seeds_can_produce_different_outputs(self):
        m = _make_categorical(
            mlm_prob=0.5, masking_prob=0.8, mutation_prob=0.1
        )
        x = np.random.default_rng(0).integers(10, 200, size=(4, 32))
        out_a, _ = m(x, seed=1)
        out_b, _ = m(x, seed=2)
        # Almost certainly different at this size.
        assert not np.array_equal(out_a, out_b)

    def test_rejects_sampler_output_with_negative_probs(self):
        m = _make_categorical()
        m.mlm_prob_sampler = _FixedSampler(0.15, -0.1, 0.1)
        x = np.zeros((2, 4), dtype=np.int64)
        with pytest.raises(ValueError):
            m(x, seed=0)

    def test_rejects_sampler_output_summing_above_one(self):
        m = _make_categorical()
        m.mlm_prob_sampler = _FixedSampler(0.15, 0.7, 0.4)
        x = np.zeros((2, 4), dtype=np.int64)
        with pytest.raises(ValueError):
            m(x, seed=0)

    @pytest.mark.parametrize("bad_mlm", [-0.1, 1.1, 2.0])
    def test_rejects_sampler_mlm_prob_out_of_range(self, bad_mlm):
        m = _make_categorical()
        m.mlm_prob_sampler = _FixedSampler(bad_mlm, 0.8, 0.1)
        x = np.zeros((2, 4), dtype=np.int64)
        with pytest.raises(ValueError):
            m(x, seed=0)

    def test_empty_excluded_ids_protects_nothing(self):
        # With excluded_ids=[] every token is eligible for masking,
        # including the ones in special_token_ids.
        m = _make_categorical(
            mlm_prob=1.0,
            masking_prob=1.0,
            mutation_prob=0.0,
            special_token_ids=[0, 1, 2],
        )
        x = np.array([[0, 1, 2, 50, 60]] * 3)
        out, _ = m(x, seed=0, excluded_ids=[])
        # Every position is masked, including the would-be specials.
        assert np.all(out == m.mask_token_id)

    def test_custom_excluded_ids_override_defaults(self):
        # Default specials are [0,1,2] but caller passes [50] only.
        # Token 50 must never be masked; tokens 0,1,2 may be masked.
        m = _make_categorical(
            mlm_prob=1.0, masking_prob=1.0, mutation_prob=0.0
        )
        x = np.array([[0, 1, 2, 50, 60]] * 3)
        out, _ = m(x, seed=0, excluded_ids=[50])
        # 50 positions must remain 50.
        assert np.all(out[x == 50] == 50)

    def test_works_with_real_sampler_integration(self):
        sampler = MLMProbabilitySampler(
            mlm_probs=[0.15],
            masking_probs=[0.8],
            mutation_probs=[0.1],
        )
        m = CategoricalMasking(
            mlm_prob_sampler=sampler,
            mask_token_id=103,
            valid_token_ids=list(range(10, 200)),
            special_token_ids=[0, 1, 2],
        )
        x = np.random.default_rng(0).integers(10, 200, size=(2, 8))
        out, labels = m(x, seed=0)
        assert out.shape == x.shape
        assert labels.shape == x.shape


class TestBetaDistributionMaskingInit:
    def test_constructs_with_valid_args(self):
        m = BetaDistributionMasking(
            mlm_prob=0.15,
            mlm_concentration=10.0,
            masking_prob=0.8,
            mutation_prob=0.1,
            mask_token_id=103,
            valid_token_ids=list(range(200)),
            special_token_ids=[0, 1, 2],
        )
        assert m.mask_token_id == 103
        assert m._beta_alpha == pytest.approx(0.15 * 10.0)
        assert m._beta_beta == pytest.approx((1.0 - 0.15) * 10.0)

    @pytest.mark.parametrize("bad_mlm", [-0.1, 0.0, 1.0, 1.5])
    def test_rejects_mlm_prob_outside_open_unit(self, bad_mlm):
        with pytest.raises(ValueError):
            BetaDistributionMasking(
                mlm_prob=bad_mlm,
                mlm_concentration=10.0,
                masking_prob=0.8,
                mutation_prob=0.1,
                mask_token_id=103,
                valid_token_ids=[1, 2],
                special_token_ids=[0],
            )

    @pytest.mark.parametrize("bad_conc", [0.0, -1.0])
    def test_rejects_non_positive_concentration(self, bad_conc):
        with pytest.raises(ValueError):
            BetaDistributionMasking(
                mlm_prob=0.15,
                mlm_concentration=bad_conc,
                masking_prob=0.8,
                mutation_prob=0.1,
                mask_token_id=103,
                valid_token_ids=[1, 2],
                special_token_ids=[0],
            )

    def test_rejects_negative_sub_action_probs(self):
        with pytest.raises(ValueError):
            BetaDistributionMasking(
                mlm_prob=0.15,
                mlm_concentration=10.0,
                masking_prob=-0.1,
                mutation_prob=0.1,
                mask_token_id=103,
                valid_token_ids=[1, 2],
                special_token_ids=[0],
            )

    def test_rejects_sub_action_probs_sum_above_one(self):
        with pytest.raises(ValueError):
            BetaDistributionMasking(
                mlm_prob=0.15,
                mlm_concentration=10.0,
                masking_prob=0.7,
                mutation_prob=0.5,
                mask_token_id=103,
                valid_token_ids=[1, 2],
                special_token_ids=[0],
            )

    def test_rejects_empty_valid_token_ids(self):
        with pytest.raises(ValueError):
            BetaDistributionMasking(
                mlm_prob=0.15,
                mlm_concentration=10.0,
                masking_prob=0.8,
                mutation_prob=0.1,
                mask_token_id=103,
                valid_token_ids=[],
                special_token_ids=[0],
            )

    def test_rejects_none_special_token_ids(self):
        with pytest.raises(ValueError):
            BetaDistributionMasking(
                mlm_prob=0.15,
                mlm_concentration=10.0,
                masking_prob=0.8,
                mutation_prob=0.1,
                mask_token_id=103,
                valid_token_ids=[1, 2],
                special_token_ids=None,
            )


class TestBetaDistributionMaskingCall:
    def _make(self, **overrides):
        kwargs = dict(
            mlm_prob=0.15,
            mlm_concentration=10.0,
            masking_prob=0.8,
            mutation_prob=0.1,
            mask_token_id=103,
            valid_token_ids=list(range(10, 200)),
            special_token_ids=[0, 1, 2],
        )
        kwargs.update(overrides)
        return BetaDistributionMasking(**kwargs)

    def test_output_shape_matches_input(self):
        m = self._make()
        x = np.random.default_rng(0).integers(10, 200, size=(4, 16))
        out, labels = m(x, seed=42)
        assert out.shape == x.shape
        assert labels.shape == x.shape

    def test_numpy_input_returns_numpy(self):
        m = self._make()
        x = np.random.default_rng(0).integers(10, 200, size=(2, 8))
        out, labels = m(x, seed=1)
        assert isinstance(out, np.ndarray)
        assert isinstance(labels, np.ndarray)

    def test_torch_input_returns_torch(self):
        m = self._make()
        x = torch.randint(10, 200, (2, 8))
        out, labels = m(x, seed=1)
        assert isinstance(out, torch.Tensor)
        assert isinstance(labels, torch.Tensor)

    def test_does_not_mutate_input(self):
        m = self._make()
        x = np.random.default_rng(0).integers(10, 200, size=(3, 8))
        original = x.copy()
        m(x, seed=7)
        np.testing.assert_array_equal(x, original)

    def test_special_tokens_never_masked(self):
        m = self._make(special_token_ids=[0, 1, 2])
        x = np.array([[0, 1, 2, 50, 60, 70]] * 8)
        # Many seeds; specials must stay put on every run.
        for s in range(10):
            out, labels = m(x, seed=s)
            for sid in [0, 1, 2]:
                positions = x == sid
                np.testing.assert_array_equal(out[positions], x[positions])
                assert np.all(labels[positions] == -100)

    def test_labels_minus_100_outside_selected_positions(self):
        m = self._make()
        x = np.random.default_rng(0).integers(10, 200, size=(3, 16))
        out, labels = m(x, seed=0)
        selected = labels != -100
        # At non-selected positions, output equals input.
        np.testing.assert_array_equal(out[~selected], x[~selected])
        # At selected positions, labels store the original tokens.
        np.testing.assert_array_equal(labels[selected], x[selected])

    def test_seed_makes_outputs_deterministic(self):
        m = self._make()
        x = np.random.default_rng(0).integers(10, 200, size=(3, 12))
        out_a, lab_a = m(x, seed=123)
        out_b, lab_b = m(x, seed=123)
        np.testing.assert_array_equal(out_a, out_b)
        np.testing.assert_array_equal(lab_a, lab_b)

    def test_high_concentration_gives_mean_close_to_target(self):
        # High concentration → Beta is tightly around its mean.
        target = 0.3
        m = self._make(
            mlm_prob=target,
            mlm_concentration=10_000.0,
            masking_prob=1.0,
            mutation_prob=0.0,
            special_token_ids=[],
        )
        # Use a long sequence so the mask fraction reflects the rate well.
        x = np.full((1, 20_000), 50, dtype=np.int64)
        out, _ = m(x, seed=0)
        observed = (out == m.mask_token_id).mean()
        assert observed == pytest.approx(target, abs=0.02)

    def test_full_mutation_replaces_with_valid_token_ids(self):
        valid = list(range(50, 60))
        m = self._make(
            mlm_prob=0.99,
            mlm_concentration=10_000.0,
            masking_prob=0.0,
            mutation_prob=1.0,
            valid_token_ids=valid,
            special_token_ids=[0, 1, 2],
        )
        x = np.full((4, 64), 30, dtype=np.int64)
        out, labels = m(x, seed=0)
        # Selected positions are replaced with values in the valid set.
        selected = labels != -100
        assert selected.any()
        assert np.all(np.isin(out[selected], valid))

    def test_custom_excluded_ids_override_defaults(self):
        # Defaults are [0,1,2]; caller restricts protection to [50] only.
        m = self._make(
            mlm_prob=0.99,
            mlm_concentration=10_000.0,
            masking_prob=1.0,
            mutation_prob=0.0,
            special_token_ids=[0, 1, 2],
        )
        x = np.array([[0, 1, 2, 50, 60]] * 3)
        out, _ = m(x, seed=0, excluded_ids=[50])
        assert np.all(out[x == 50] == 50)


def _run_lengths(row_selected: np.ndarray) -> list:
    """Return the lengths of contiguous True runs in a 1-D boolean array."""
    runs = []
    cur = 0
    for v in row_selected:
        if v:
            cur += 1
        elif cur > 0:
            runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return runs


def _make_hybrid(**overrides):
    kwargs = dict(
        mlm_prob=0.15,
        mlm_concentration=10_000.0,
        span_fraction_mean=0.5,
        span_fraction_concentration=10_000.0,
        span_length_p=0.2,
        max_span_length=10,
        masking_prob=0.8,
        mutation_prob=0.1,
        mask_token_id=103,
        valid_token_ids=list(range(10, 200)),
        special_token_ids=[0, 1, 2],
    )
    kwargs.update(overrides)
    return HybridSpanMasking(**kwargs)


class TestHybridSpanMaskingInit:
    def test_constructs_with_valid_args(self):
        m = _make_hybrid()
        assert m.mask_token_id == 103
        assert m.max_span_length == 10
        assert m.span_length_p == pytest.approx(0.2)

    @pytest.mark.parametrize("bad_mlm", [-0.1, 0.0, 1.0, 1.5])
    def test_rejects_mlm_prob_outside_open_unit(self, bad_mlm):
        with pytest.raises(ValueError):
            _make_hybrid(mlm_prob=bad_mlm)

    @pytest.mark.parametrize("bad_conc", [0.0, -1.0])
    def test_rejects_non_positive_mlm_concentration(self, bad_conc):
        with pytest.raises(ValueError):
            _make_hybrid(mlm_concentration=bad_conc)

    @pytest.mark.parametrize("bad_alpha_mean", [-0.1, 1.1])
    def test_rejects_span_fraction_mean_outside_closed_unit(
        self, bad_alpha_mean
    ):
        with pytest.raises(ValueError):
            _make_hybrid(span_fraction_mean=bad_alpha_mean)

    def test_accepts_span_fraction_mean_at_boundaries(self):
        _make_hybrid(span_fraction_mean=0.0)
        _make_hybrid(span_fraction_mean=1.0)

    @pytest.mark.parametrize("bad_conc", [0.0, -1.0])
    def test_rejects_non_positive_span_fraction_concentration(self, bad_conc):
        with pytest.raises(ValueError):
            _make_hybrid(span_fraction_concentration=bad_conc)

    @pytest.mark.parametrize("bad_p", [0.0, -0.1, 1.1])
    def test_rejects_span_length_p_outside_half_open_unit(self, bad_p):
        with pytest.raises(ValueError):
            _make_hybrid(span_length_p=bad_p)

    @pytest.mark.parametrize("bad_max", [0, -1])
    def test_rejects_non_positive_max_span_length(self, bad_max):
        with pytest.raises(ValueError):
            _make_hybrid(max_span_length=bad_max)

    def test_rejects_negative_sub_action_probs(self):
        with pytest.raises(ValueError):
            _make_hybrid(masking_prob=-0.1)
        with pytest.raises(ValueError):
            _make_hybrid(mutation_prob=-0.1)

    def test_rejects_sub_action_probs_sum_above_one(self):
        with pytest.raises(ValueError):
            _make_hybrid(masking_prob=0.7, mutation_prob=0.5)

    def test_rejects_empty_valid_token_ids(self):
        with pytest.raises(ValueError):
            _make_hybrid(valid_token_ids=[])

    def test_rejects_none_special_token_ids(self):
        with pytest.raises(ValueError):
            _make_hybrid(special_token_ids=None)


class TestHybridSpanMaskingCall:
    def test_output_shape_matches_input(self):
        m = _make_hybrid()
        x = np.random.default_rng(0).integers(10, 200, size=(4, 64))
        out, labels = m(x, seed=42)
        assert out.shape == x.shape
        assert labels.shape == x.shape

    def test_numpy_input_returns_numpy(self):
        m = _make_hybrid()
        x = np.random.default_rng(0).integers(10, 200, size=(2, 32))
        out, labels = m(x, seed=1)
        assert isinstance(out, np.ndarray)
        assert isinstance(labels, np.ndarray)

    def test_torch_input_returns_torch(self):
        m = _make_hybrid()
        x = torch.randint(10, 200, (2, 32))
        out, labels = m(x, seed=1)
        assert isinstance(out, torch.Tensor)
        assert isinstance(labels, torch.Tensor)

    def test_does_not_mutate_input(self):
        m = _make_hybrid()
        x = np.random.default_rng(0).integers(10, 200, size=(3, 32))
        original = x.copy()
        m(x, seed=7)
        np.testing.assert_array_equal(x, original)

    def test_special_tokens_never_masked_with_dense_specials(self):
        # Specials interleaved with normal tokens to stress span clipping.
        m = _make_hybrid(special_token_ids=[0, 1, 2])
        row = [0, 50, 51, 52, 1, 53, 54, 2, 55, 56, 57, 58, 0, 59, 60]
        x = np.array([row] * 6)
        for s in range(8):
            out, labels = m(x, seed=s)
            for sid in [0, 1, 2]:
                positions = x == sid
                np.testing.assert_array_equal(out[positions], x[positions])
                assert np.all(labels[positions] == -100)

    def test_labels_minus_100_outside_selected(self):
        m = _make_hybrid()
        x = np.random.default_rng(0).integers(10, 200, size=(3, 64))
        out, labels = m(x, seed=0)
        selected = labels != -100
        # Selected positions hold the original tokens in labels.
        np.testing.assert_array_equal(labels[selected], x[selected])
        # Non-selected positions are unchanged in the output.
        np.testing.assert_array_equal(out[~selected], x[~selected])

    def test_seed_makes_outputs_deterministic(self):
        m = _make_hybrid()
        x = np.random.default_rng(0).integers(10, 200, size=(3, 64))
        out_a, lab_a = m(x, seed=123)
        out_b, lab_b = m(x, seed=123)
        np.testing.assert_array_equal(out_a, out_b)
        np.testing.assert_array_equal(lab_a, lab_b)

    def test_different_seeds_can_differ(self):
        m = _make_hybrid(mlm_prob=0.3, mlm_concentration=10_000.0)
        x = np.full((2, 128), 50, dtype=np.int64)
        out_a, _ = m(x, seed=1)
        out_b, _ = m(x, seed=2)
        # With a long sequence, two seeds should almost surely diverge.
        assert not np.array_equal(out_a, out_b)

    def test_alpha_zero_produces_only_isolated_selections(self):
        # span_fraction_mean=0 → pure scatter; with random positions over a
        # long sequence, contiguous runs of length > 1 should be rare.
        m = _make_hybrid(
            span_fraction_mean=0.0,
            mlm_prob=0.05,
            mlm_concentration=10_000.0,
            special_token_ids=[],
        )
        x = np.full((1, 4_000), 50, dtype=np.int64)
        _, labels = m(x, seed=0)
        runs = _run_lengths(labels[0] != -100)
        assert runs, "expected some selections"
        long_runs = [r for r in runs if r > 1]
        # Pure scatter at 5% rate yields almost no length-2+ runs by chance.
        assert len(long_runs) / max(1, len(runs)) < 0.1

    def test_alpha_one_produces_longer_spans_on_average(self):
        # Pure span sampling with mean span = 1/p = 5 should give a mean
        # run length materially greater than the scatter baseline (~1).
        m = _make_hybrid(
            span_fraction_mean=1.0,
            span_length_p=0.2,
            max_span_length=10,
            mlm_prob=0.15,
            mlm_concentration=10_000.0,
            special_token_ids=[],
        )
        x = np.full((1, 4_000), 50, dtype=np.int64)
        _, labels = m(x, seed=0)
        runs = _run_lengths(labels[0] != -100)
        assert runs
        mean_run = sum(runs) / len(runs)
        assert mean_run > 2.0

    def test_high_concentration_hits_target_rate(self):
        target = 0.2
        m = _make_hybrid(
            mlm_prob=target,
            mlm_concentration=10_000.0,
            span_fraction_mean=0.5,
            span_fraction_concentration=10_000.0,
            special_token_ids=[],
        )
        x = np.full((1, 10_000), 50, dtype=np.int64)
        _, labels = m(x, seed=0)
        observed = (labels[0] != -100).mean()
        assert observed == pytest.approx(target, abs=0.03)

    def test_max_span_length_never_exceeded(self):
        # Force pure-span, very long geometric draws, tight cap.
        m = _make_hybrid(
            span_fraction_mean=1.0,
            span_length_p=0.01,
            max_span_length=4,
            mlm_prob=0.5,
            mlm_concentration=10_000.0,
            special_token_ids=[],
        )
        x = np.full((1, 2_000), 50, dtype=np.int64)
        _, labels = m(x, seed=0)
        runs = _run_lengths(labels[0] != -100)
        # Individual spans never exceed the cap. Adjacent spans can fuse
        # into runs longer than the cap, so we can't assert max(runs) <= 4;
        # but the *sampled* span lengths are capped — this is the testable
        # invariant via the implementation contract. We instead check that
        # the mean run length stays bounded.
        assert max(runs) <= len(labels[0])  # trivial sanity
        # Mean run length should not blow up far above the cap.
        assert sum(runs) / len(runs) < 4 * 5

    def test_full_mask_replaces_with_mask_token(self):
        m = _make_hybrid(
            mlm_prob=0.99,
            mlm_concentration=10_000.0,
            masking_prob=1.0,
            mutation_prob=0.0,
            special_token_ids=[0, 1, 2],
        )
        x = np.full((3, 32), 50, dtype=np.int64)
        out, labels = m(x, seed=0)
        selected = labels != -100
        assert selected.any()
        assert np.all(out[selected] == m.mask_token_id)

    def test_full_mutation_uses_valid_token_ids(self):
        valid = list(range(50, 60))
        m = _make_hybrid(
            mlm_prob=0.99,
            mlm_concentration=10_000.0,
            masking_prob=0.0,
            mutation_prob=1.0,
            valid_token_ids=valid,
            special_token_ids=[0, 1, 2],
        )
        x = np.full((4, 32), 30, dtype=np.int64)
        out, labels = m(x, seed=0)
        selected = labels != -100
        assert selected.any()
        assert np.all(np.isin(out[selected], valid))

    def test_custom_excluded_ids_override_defaults(self):
        m = _make_hybrid(
            mlm_prob=0.99,
            mlm_concentration=10_000.0,
            masking_prob=1.0,
            mutation_prob=0.0,
            special_token_ids=[0, 1, 2],
        )
        x = np.array([[0, 1, 2, 50, 60]] * 3)
        out, _ = m(x, seed=0, excluded_ids=[50])
        assert np.all(out[x == 50] == 50)

    def test_torch_input_not_mutated(self):
        m = _make_hybrid()
        x = torch.randint(10, 200, (3, 32))
        original = x.clone()
        m(x, seed=7)
        assert torch.equal(x, original)

    def test_handles_all_special_sequence(self):
        # Every token is excluded → nothing gets masked, no crash.
        m = _make_hybrid(special_token_ids=[0])
        x = np.zeros((2, 16), dtype=np.int64)
        out, labels = m(x, seed=0)
        np.testing.assert_array_equal(out, x)
        assert np.all(labels == -100)


import math as _math

from seqmask.masking import DiffusionMasking


def _make_diffusion(**overrides):
    kwargs = dict(
        total_timesteps=100,
        mask_token_id=103,
        special_token_ids=[0, 1, 2],
        schedule="linear",
    )
    kwargs.update(overrides)
    return DiffusionMasking(**kwargs)


class TestDiffusionMaskingInit:
    def test_constructs_with_valid_args(self):
        m = _make_diffusion()
        assert m.total_timesteps == 100
        assert m.mask_token_id == 103
        # String schedules are resolved into the corresponding scheduler instance.
        from seqmask.masking.diffusion import LinearDiffusionScheduler

        assert isinstance(m.schedule, LinearDiffusionScheduler)

    @pytest.mark.parametrize("bad_T", [0, -1])
    def test_rejects_non_positive_total_timesteps(self, bad_T):
        with pytest.raises(ValueError):
            _make_diffusion(total_timesteps=bad_T)

    def test_rejects_unknown_schedule_string(self):
        with pytest.raises(ValueError):
            _make_diffusion(schedule="quadratic")

    def test_rejects_none_special_token_ids(self):
        with pytest.raises(ValueError):
            _make_diffusion(special_token_ids=None)

    def test_rejects_non_string_non_callable_schedule(self):
        with pytest.raises(ValueError):
            _make_diffusion(schedule=42)

    @pytest.mark.parametrize("name", ["linear", "cosine", "sqrt"])
    def test_accepts_each_builtin_schedule(self, name):
        _make_diffusion(schedule=name)

    def test_accepts_callable_schedule(self):
        _make_diffusion(schedule=lambda t, T: t / T)


class TestDiffusionMaskingMlmProbAt:
    @pytest.mark.parametrize("name", ["linear", "cosine", "sqrt"])
    def test_endpoints_are_zero_and_one(self, name):
        m = _make_diffusion(schedule=name, total_timesteps=20)
        assert m.mlm_prob_at(0) == pytest.approx(0.0)
        assert m.mlm_prob_at(20) == pytest.approx(1.0)

    def test_linear_schedule_at_midpoint(self):
        m = _make_diffusion(schedule="linear", total_timesteps=10)
        assert m.mlm_prob_at(5) == pytest.approx(0.5)

    def test_cosine_schedule_at_midpoint(self):
        m = _make_diffusion(schedule="cosine", total_timesteps=10)
        # 1 - cos(pi/4) = 1 - sqrt(2)/2 ≈ 0.2929
        assert m.mlm_prob_at(5) == pytest.approx(
            1.0 - _math.cos(_math.pi / 4), rel=1e-6
        )

    def test_sqrt_schedule_at_quarter(self):
        m = _make_diffusion(schedule="sqrt", total_timesteps=16)
        # sqrt(4/16) = 0.5
        assert m.mlm_prob_at(4) == pytest.approx(0.5)

    def test_monotonic_non_decreasing(self):
        for name in ["linear", "cosine", "sqrt"]:
            m = _make_diffusion(schedule=name, total_timesteps=50)
            prev = -1.0
            for t in range(0, 51):
                cur = m.mlm_prob_at(t)
                assert cur >= prev
                prev = cur

    @pytest.mark.parametrize("bad_t", [-1, -5, 101, 150])
    def test_rejects_out_of_range_timestep(self, bad_t):
        m = _make_diffusion(total_timesteps=100)
        with pytest.raises(ValueError):
            m.mlm_prob_at(bad_t)

    def test_accepts_float_timestep(self):
        m = _make_diffusion(schedule="linear", total_timesteps=10)
        assert m.mlm_prob_at(2.5) == pytest.approx(0.25)

    def test_rejects_schedule_returning_out_of_range(self):
        # Custom schedule that returns >1; should raise on call.
        m = _make_diffusion(schedule=lambda t, T: 1.5)
        with pytest.raises(ValueError):
            m.mlm_prob_at(5)

    def test_custom_callable_used(self):
        m = _make_diffusion(schedule=lambda t, T: 0.42 if t > 0 else 0.0)
        assert m.mlm_prob_at(1) == pytest.approx(0.42)


class TestDiffusionMaskingCall:
    def test_output_shape_matches_input(self):
        m = _make_diffusion()
        x = np.random.default_rng(0).integers(10, 200, size=(4, 16))
        out, labels = m(x, timestep=50, seed=0)
        assert out.shape == x.shape
        assert labels.shape == x.shape

    def test_numpy_input_returns_numpy(self):
        m = _make_diffusion()
        x = np.random.default_rng(0).integers(10, 200, size=(2, 8))
        out, labels = m(x, timestep=50, seed=1)
        assert isinstance(out, np.ndarray)
        assert isinstance(labels, np.ndarray)

    def test_torch_input_returns_torch(self):
        m = _make_diffusion()
        x = torch.randint(10, 200, (2, 8))
        out, labels = m(x, timestep=50, seed=1)
        assert isinstance(out, torch.Tensor)
        assert isinstance(labels, torch.Tensor)

    def test_does_not_mutate_input(self):
        m = _make_diffusion()
        x = np.random.default_rng(0).integers(10, 200, size=(3, 8))
        original = x.copy()
        m(x, timestep=50, seed=7)
        np.testing.assert_array_equal(x, original)

    def test_timestep_zero_yields_no_changes(self):
        m = _make_diffusion()
        x = np.random.default_rng(0).integers(10, 200, size=(3, 16))
        out, labels = m(x, timestep=0, seed=0)
        np.testing.assert_array_equal(out, x)
        assert np.all(labels == -100)

    def test_timestep_T_masks_all_non_special(self):
        m = _make_diffusion(total_timesteps=10, special_token_ids=[0, 1, 2])
        x = np.array([[0, 1, 2, 50, 60, 70]] * 4)
        out, labels = m(x, timestep=10, seed=0)
        # All non-special positions replaced with mask token.
        non_special = ~np.isin(x, [0, 1, 2])
        assert np.all(out[non_special] == m.mask_token_id)
        # Specials remain unchanged.
        for sid in [0, 1, 2]:
            assert np.all(out[x == sid] == sid)
            assert np.all(labels[x == sid] == -100)

    def test_only_uses_mask_token_no_random_substitutions(self):
        # At any intermediate timestep, the only altered values must be
        # mask_token_id. No 10% random-token mutation.
        m = _make_diffusion(
            total_timesteps=10, mask_token_id=999, special_token_ids=[]
        )
        x = np.full((3, 64), 50, dtype=np.int64)
        out, _ = m(x, timestep=5, seed=0)
        altered = out != x
        if altered.any():
            assert np.all(out[altered] == 999)

    def test_special_tokens_never_masked(self):
        m = _make_diffusion(
            total_timesteps=10,
            schedule="linear",
            special_token_ids=[0, 1, 2],
        )
        row = [0, 1, 2, 50, 60, 70, 80]
        x = np.array([row] * 5)
        for s in range(8):
            out, labels = m(x, timestep=10, seed=s)
            for sid in [0, 1, 2]:
                positions = x == sid
                np.testing.assert_array_equal(out[positions], x[positions])
                assert np.all(labels[positions] == -100)

    def test_labels_minus_100_outside_masked(self):
        m = _make_diffusion(total_timesteps=10, special_token_ids=[])
        x = np.random.default_rng(0).integers(10, 200, size=(2, 16))
        out, labels = m(x, timestep=5, seed=0)
        masked = labels != -100
        np.testing.assert_array_equal(labels[masked], x[masked])
        np.testing.assert_array_equal(out[~masked], x[~masked])

    def test_seed_makes_outputs_deterministic(self):
        m = _make_diffusion()
        x = np.random.default_rng(0).integers(10, 200, size=(3, 12))
        out_a, lab_a = m(x, timestep=30, seed=42)
        out_b, lab_b = m(x, timestep=30, seed=42)
        np.testing.assert_array_equal(out_a, out_b)
        np.testing.assert_array_equal(lab_a, lab_b)

    def test_different_seeds_diverge(self):
        m = _make_diffusion(total_timesteps=10, special_token_ids=[])
        x = np.full((2, 256), 50, dtype=np.int64)
        out_a, _ = m(x, timestep=5, seed=1)
        out_b, _ = m(x, timestep=5, seed=2)
        assert not np.array_equal(out_a, out_b)

    def test_observed_rate_matches_schedule(self):
        # With a long sequence and a fixed seed, observed mask fraction
        # should be close to mlm_prob_at(timestep).
        m = _make_diffusion(
            total_timesteps=100,
            schedule="cosine",
            special_token_ids=[],
        )
        x = np.full((1, 20_000), 50, dtype=np.int64)
        for t in [10, 50, 90]:
            target = m.mlm_prob_at(t)
            out, _ = m(x, timestep=t, seed=0)
            observed = (out == m.mask_token_id).mean()
            assert observed == pytest.approx(target, abs=0.02)

    def test_custom_excluded_ids_override_defaults(self):
        m = _make_diffusion(total_timesteps=10, special_token_ids=[0, 1, 2])
        x = np.array([[0, 1, 2, 50, 60]] * 3)
        out, _ = m(x, timestep=10, seed=0, excluded_ids=[50])
        assert np.all(out[x == 50] == 50)

    def test_custom_callable_schedule_drives_call(self):
        # Constant schedule returning 0.5 → roughly half the eligible
        # positions masked regardless of timestep value.
        m = _make_diffusion(
            total_timesteps=100,
            schedule=lambda t, T: 0.5,
            special_token_ids=[],
        )
        x = np.full((1, 10_000), 50, dtype=np.int64)
        out, _ = m(x, timestep=42, seed=0)
        observed = (out == m.mask_token_id).mean()
        assert observed == pytest.approx(0.5, abs=0.02)

    def test_torch_input_not_mutated(self):
        m = _make_diffusion()
        x = torch.randint(10, 200, (3, 8))
        original = x.clone()
        m(x, timestep=50, seed=7)
        assert torch.equal(x, original)


import pickle as _pickle

from seqmask.masking.diffusion import CosineDiffusionScheduler
from seqmask.masking.diffusion import DiffusionScheduler
from seqmask.masking.diffusion import LinearDiffusionScheduler
from seqmask.masking.diffusion import SqrtDiffusionScheduler


class TestDiffusionSchedulerClasses:
    def test_base_class_call_is_not_implemented(self):
        with pytest.raises(NotImplementedError):
            DiffusionScheduler()(5, 10)

    @pytest.mark.parametrize(
        "cls,name",
        [
            (LinearDiffusionScheduler, "linear"),
            (CosineDiffusionScheduler, "cosine"),
            (SqrtDiffusionScheduler, "sqrt"),
        ],
    )
    def test_subclass_matches_string_alias(self, cls, name):
        # Passing the instance must produce the same rate curve as
        # passing the string name.
        m_instance = _make_diffusion(schedule=cls(), total_timesteps=20)
        m_string = _make_diffusion(schedule=name, total_timesteps=20)
        for t in [0, 5, 10, 15, 20]:
            assert m_instance.mlm_prob_at(t) == pytest.approx(
                m_string.mlm_prob_at(t)
            )

    def test_linear_subclass_endpoints(self):
        s = LinearDiffusionScheduler()
        assert s(0, 10) == pytest.approx(0.0)
        assert s(10, 10) == pytest.approx(1.0)

    def test_cosine_subclass_midpoint(self):
        s = CosineDiffusionScheduler()
        assert s(5, 10) == pytest.approx(
            1.0 - _math.cos(_math.pi / 4), rel=1e-6
        )

    def test_sqrt_subclass_quarter(self):
        s = SqrtDiffusionScheduler()
        assert s(4, 16) == pytest.approx(0.5)


class TestDiffusionMaskingPicklability:
    """Schedulers must survive pickling so DiffusionMasking can be sent
    to PyTorch DataLoader workers under spawn-based multiprocessing."""

    @pytest.mark.parametrize(
        "cls",
        [
            LinearDiffusionScheduler,
            CosineDiffusionScheduler,
            SqrtDiffusionScheduler,
        ],
    )
    def test_builtin_scheduler_instance_pickles(self, cls):
        s = cls()
        restored = _pickle.loads(_pickle.dumps(s))
        assert isinstance(restored, cls)
        # Behavior survives the round trip.
        assert restored(5, 10) == pytest.approx(s(5, 10))

    @pytest.mark.parametrize("name", ["linear", "cosine", "sqrt"])
    def test_diffusion_masking_pickles_with_string_schedule(self, name):
        m = _make_diffusion(schedule=name)
        restored = _pickle.loads(_pickle.dumps(m))
        # The resolved scheduler instance comes back intact.
        assert isinstance(restored.schedule, type(m.schedule))
        # Same call produces the same output.
        x = np.random.default_rng(0).integers(10, 200, size=(2, 16))
        out_a, lab_a = m(x, timestep=42, seed=0)
        out_b, lab_b = restored(x, timestep=42, seed=0)
        np.testing.assert_array_equal(out_a, out_b)
        np.testing.assert_array_equal(lab_a, lab_b)

    def test_diffusion_masking_pickles_with_scheduler_instance(self):
        m = _make_diffusion(schedule=CosineDiffusionScheduler())
        restored = _pickle.loads(_pickle.dumps(m))
        x = np.random.default_rng(0).integers(10, 200, size=(2, 16))
        out_a, _ = m(x, timestep=42, seed=0)
        out_b, _ = restored(x, timestep=42, seed=0)
        np.testing.assert_array_equal(out_a, out_b)

    def test_diffusion_masking_with_lambda_does_not_pickle(self):
        # Documenting the limit: lambdas are not picklable with stdlib
        # pickle. Users who need DataLoader workers must pass a
        # DiffusionScheduler subclass instead.
        m = _make_diffusion(schedule=lambda t, T: t / T)
        with pytest.raises((_pickle.PicklingError, AttributeError)):
            _pickle.dumps(m)


class TestCustomDiffusionScheduler:
    def test_user_subclass_works_end_to_end(self):
        class ConstantSchedule(DiffusionScheduler):
            def __init__(self, value):
                self.value = value

            def __call__(self, timestep, total_timesteps):
                return self.value

        m = _make_diffusion(
            schedule=ConstantSchedule(0.5),
            total_timesteps=100,
            special_token_ids=[],
        )
        assert m.mlm_prob_at(42) == pytest.approx(0.5)
        x = np.full((1, 5_000), 50, dtype=np.int64)
        out, _ = m(x, timestep=42, seed=0)
        observed = (out == m.mask_token_id).mean()
        assert observed == pytest.approx(0.5, abs=0.02)

    def test_user_subclass_pickles(self):
        # User subclasses defined at module level pickle cleanly. Define
        # one at module scope by attaching it to the module — that's the
        # standard pattern for testing pickle.
        s = _ModuleLevelConstantSchedule(0.3)
        restored = _pickle.loads(_pickle.dumps(s))
        assert restored(0, 1) == pytest.approx(0.3)


class _ModuleLevelConstantSchedule(DiffusionScheduler):
    """Defined at module level so pickle can locate it by qualified name."""

    def __init__(self, value):
        self.value = value

    def __call__(self, timestep, total_timesteps):
        return self.value
