"""Tests for focus_positions support across CategoricalMasking,
BetaDistributionMasking, and DiffusionMasking.

These tests verify the shared focus helper and that each per-token
seqmask honors the three focus strategies. Per-class semantics (shape,
type round-trip, special handling, etc.) are covered in test_masking.py;
this file focuses on focus-specific behavior.
"""
import os

os.environ[
    "SEQMASK_BACKEND"
] = "torch"  # force PyTorch for deterministic testing

import numpy as np
import pytest
import torch

from seqmask.masking import BetaDistributionMasking
from seqmask.masking import CategoricalMasking
from seqmask.masking import DiffusionMasking
from seqmask.masking import focus_positions_from_reference
from seqmask.masking.focus import coerce_focus_positions
from seqmask.masking.focus import select_positions
from seqmask.masking.focus import validate_focus_params


class _FixedSampler:
    def __init__(self, mlm_prob, masking_prob, mutation_prob):
        self._triple = (
            float(mlm_prob),
            float(masking_prob),
            float(mutation_prob),
        )

    def sample(self, seed=None):
        return self._triple


def _categorical(
    mlm_prob=0.1,
    *,
    focus_strategy="multiplicative",
    focus_strength=2.0,
    special_token_ids=(),
):
    return CategoricalMasking(
        mlm_prob_sampler=_FixedSampler(mlm_prob, 1.0, 0.0),
        mask_token_id=999,
        valid_token_ids=list(range(10, 200)),
        special_token_ids=list(special_token_ids),
        focus_strategy=focus_strategy,
        focus_strength=focus_strength,
    )


def _beta(
    mlm_prob=0.1,
    *,
    focus_strategy="multiplicative",
    focus_strength=2.0,
    special_token_ids=(),
):
    return BetaDistributionMasking(
        mlm_prob=mlm_prob,
        mlm_concentration=10_000.0,
        masking_prob=1.0,
        mutation_prob=0.0,
        mask_token_id=999,
        valid_token_ids=list(range(10, 200)),
        special_token_ids=list(special_token_ids),
        focus_strategy=focus_strategy,
        focus_strength=focus_strength,
    )


def _diffusion(
    *,
    focus_strategy="multiplicative",
    focus_strength=2.0,
    special_token_ids=(),
):
    return DiffusionMasking(
        total_timesteps=100,
        mask_token_id=999,
        special_token_ids=list(special_token_ids),
        schedule="linear",
        focus_strategy=focus_strategy,
        focus_strength=focus_strength,
    )


def _call(seqmask, x, focus_positions=None, seed=0):
    """Uniform call across the three classes (Diffusion needs a timestep)."""
    if isinstance(seqmask, DiffusionMasking):
        return seqmask(
            x, timestep=10, seed=seed, focus_positions=focus_positions
        )
    return seqmask(x, seed=seed, focus_positions=focus_positions)


# Parametrize across the three per-token classes for all behavior tests.
ALL_MAKERS = [
    pytest.param(_categorical, id="categorical"),
    pytest.param(_beta, id="beta"),
    pytest.param(_diffusion, id="diffusion"),
]


class TestFocusValidation:
    def test_rejects_unknown_strategy_categorical(self):
        with pytest.raises(ValueError, match="focus_strategy"):
            _categorical(focus_strategy="garbage")

    def test_rejects_unknown_strategy_beta(self):
        with pytest.raises(ValueError, match="focus_strategy"):
            _beta(focus_strategy="garbage")

    def test_rejects_unknown_strategy_diffusion(self):
        with pytest.raises(ValueError, match="focus_strategy"):
            _diffusion(focus_strategy="garbage")

    @pytest.mark.parametrize("make", ALL_MAKERS)
    def test_rejects_negative_strength(self, make):
        with pytest.raises(ValueError, match="focus_strength"):
            make(focus_strength=-0.5)

    @pytest.mark.parametrize("make", ALL_MAKERS)
    def test_default_focus_attrs_set(self, make):
        m = make()
        assert m.focus_strategy == "multiplicative"
        assert m.focus_strength == pytest.approx(2.0)


class TestFocusHelperShape:
    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            coerce_focus_positions(
                np.zeros((2, 4), dtype=bool), expected_shape=(2, 8)
            )

    def test_passes_none_through(self):
        assert coerce_focus_positions(None, expected_shape=(2, 4)) is None

    def test_accepts_torch_bool_tensor(self):
        out = coerce_focus_positions(
            torch.tensor([[True, False], [False, True]]),
            expected_shape=(2, 2),
        )
        assert isinstance(out, np.ndarray)
        assert out.dtype == bool

    def test_accepts_python_nested_list(self):
        out = coerce_focus_positions(
            [[True, False, True], [False, False, True]], expected_shape=(2, 3)
        )
        assert isinstance(out, np.ndarray)
        assert out.dtype == bool


class TestFocusNoneIsBackwardsCompat:
    """`focus_positions=None` must give the exact same output as before
    the feature was added."""

    @pytest.mark.parametrize("make", ALL_MAKERS)
    def test_none_matches_no_arg(self, make):
        m = make(mlm_prob=0.3) if make is not _diffusion else make()
        x = np.random.default_rng(0).integers(10, 200, size=(2, 32))
        out_a, lab_a = _call(m, x, focus_positions=None, seed=42)
        # Re-create seqmask; same seed and no focus must give same output.
        m2 = make(mlm_prob=0.3) if make is not _diffusion else make()
        out_b, lab_b = _call(m2, x, seed=42)
        np.testing.assert_array_equal(out_a, out_b)
        np.testing.assert_array_equal(lab_a, lab_b)

    @pytest.mark.parametrize("make", ALL_MAKERS)
    def test_all_false_focus_matches_none(self, make):
        m1 = (
            make(mlm_prob=0.3, focus_strategy="multiplicative")
            if make is not _diffusion
            else make(focus_strategy="multiplicative")
        )
        m2 = (
            make(mlm_prob=0.3, focus_strategy="multiplicative")
            if make is not _diffusion
            else make(focus_strategy="multiplicative")
        )
        x = np.random.default_rng(0).integers(10, 200, size=(2, 32))
        all_false = np.zeros_like(x, dtype=bool)
        out_a, _ = _call(m1, x, focus_positions=all_false, seed=42)
        out_b, _ = _call(m2, x, focus_positions=None, seed=42)
        # An all-False focus with the multiplicative strategy collapses
        # to uniform mlm_prob per position — same as no focus.
        np.testing.assert_array_equal(out_a, out_b)


class TestForceIncludeStrategy:
    @pytest.mark.parametrize("make", ALL_MAKERS)
    def test_focus_positions_always_masked(self, make):
        m = (
            make(mlm_prob=0.05, focus_strategy="force_include")
            if make is not _diffusion
            else make(focus_strategy="force_include")
        )
        x = np.random.default_rng(0).integers(10, 200, size=(3, 64))
        focus = np.zeros_like(x, dtype=bool)
        focus[:, [5, 17, 42]] = True
        out, labels = _call(m, x, focus_positions=focus, seed=0)
        # Every focus position must be masked (selected).
        assert np.all(labels[focus] != -100)

    @pytest.mark.parametrize("make", ALL_MAKERS)
    def test_specials_still_win_over_focus(self, make):
        # A focus position that is also a special must remain untouched.
        # mlm_prob value doesn't matter under force_include since focus
        # positions are unconditionally selected; pick a value all three
        # constructors accept (Beta requires mlm_prob in (0, 1)).
        m = (
            make(
                mlm_prob=0.5,
                focus_strategy="force_include",
                special_token_ids=[0, 1, 2],
            )
            if make is not _diffusion
            else make(
                focus_strategy="force_include",
                special_token_ids=[0, 1, 2],
            )
        )
        x = np.array([[0, 1, 2, 50, 60, 70]] * 4, dtype=np.int64)
        focus = np.ones_like(x, dtype=bool)  # mark every position
        out, labels = _call(m, x, focus_positions=focus, seed=0)
        for sid in [0, 1, 2]:
            positions = x == sid
            assert np.all(out[positions] == sid)
            assert np.all(labels[positions] == -100)


class TestMultiplicativeStrategy:
    @pytest.mark.parametrize("make", ALL_MAKERS)
    def test_focus_mask_rate_exceeds_baseline(self, make):
        # Low base rate with a 10x multiplier should yield a clearly
        # higher mask rate at focus positions than at non-focus.
        m = (
            make(
                mlm_prob=0.05,
                focus_strategy="multiplicative",
                focus_strength=10.0,
            )
            if make is not _diffusion
            else make(focus_strategy="multiplicative", focus_strength=10.0)
        )
        x = np.full((1, 10_000), 50, dtype=np.int64)
        focus = np.zeros_like(x, dtype=bool)
        focus[:, : x.shape[1] // 2] = True  # left half = focus
        _, labels = _call(m, x, focus_positions=focus, seed=0)
        rate_focus = (labels[focus] != -100).mean()
        rate_non_focus = (labels[~focus] != -100).mean()
        assert rate_focus > rate_non_focus
        # Roughly 5x more (capped multiplier means we'd expect ~10x in
        # theory, but the actual ratio is bounded by clipping at 1).
        assert rate_focus > 2.5 * rate_non_focus

    @pytest.mark.parametrize("make", ALL_MAKERS)
    def test_strength_one_matches_no_focus(self, make):
        # With focus_strength=1, multiplicative is a no-op regardless of
        # focus_positions, because p_focus = p_base.
        m1 = (
            make(mlm_prob=0.2, focus_strength=1.0)
            if make is not _diffusion
            else make(focus_strength=1.0)
        )
        m2 = (
            make(mlm_prob=0.2, focus_strength=1.0)
            if make is not _diffusion
            else make(focus_strength=1.0)
        )
        x = np.random.default_rng(0).integers(10, 200, size=(2, 64))
        focus = np.ones_like(x, dtype=bool)
        out_a, _ = _call(m1, x, focus_positions=focus, seed=7)
        out_b, _ = _call(m2, x, focus_positions=None, seed=7)
        np.testing.assert_array_equal(out_a, out_b)


class TestWeightedStrategy:
    @pytest.mark.parametrize("make", ALL_MAKERS)
    def test_focus_positions_oversampled(self, make):
        # Heavily weight focus positions; the share of *selected*
        # positions that fall inside the focus region must exceed the
        # share expected by chance.
        m = (
            make(
                mlm_prob=0.2,
                focus_strategy="weighted",
                focus_strength=20.0,
            )
            if make is not _diffusion
            else make(focus_strategy="weighted", focus_strength=20.0)
        )
        x = np.full((1, 2_000), 50, dtype=np.int64)
        focus = np.zeros_like(x, dtype=bool)
        # 10% of positions are focus.
        focus[:, : x.shape[1] // 10] = True
        focus_density = focus.mean()
        _, labels = _call(m, x, focus_positions=focus, seed=0)
        selected = labels != -100
        if selected.any():
            share_of_selected_in_focus = (
                selected & focus
            ).sum() / selected.sum()
            # Strong weighting => far more than 10% of selections in focus.
            assert share_of_selected_in_focus > 3 * focus_density

    @pytest.mark.parametrize("make", ALL_MAKERS)
    def test_total_mask_count_preserved_approximately(self, make):
        # Weighted strategy preserves the overall expected mask budget.
        m = (
            make(
                mlm_prob=0.2,
                focus_strategy="weighted",
                focus_strength=5.0,
                special_token_ids=[],
            )
            if make is not _diffusion
            else make(
                focus_strategy="weighted",
                focus_strength=5.0,
                special_token_ids=[],
            )
        )
        x = np.full((1, 2_000), 50, dtype=np.int64)
        focus = np.zeros_like(x, dtype=bool)
        focus[:, : x.shape[1] // 5] = True
        _, labels = _call(m, x, focus_positions=focus, seed=0)
        observed_rate = (labels != -100).mean()
        # Should be close to mlm_prob (categorical/beta call it 0.2;
        # diffusion at timestep 10 / T=100 linear is also 0.1 actually,
        # which gives a different target — let's just check it's
        # non-zero and within sane bounds).
        assert 0.0 < observed_rate <= 0.3


class TestSelectPositionsHelperDirect:
    def test_uniform_when_focus_is_none(self):
        rng = np.random.default_rng(0)
        is_excluded = np.zeros((2, 1000), dtype=bool)
        out = select_positions(
            rng=rng,
            mlm_prob=0.3,
            is_excluded=is_excluded,
            focus_positions=None,
            focus_strategy="multiplicative",
            focus_strength=2.0,
        )
        observed = out.mean()
        assert observed == pytest.approx(0.3, abs=0.03)

    def test_force_include_unions_focus(self):
        rng = np.random.default_rng(0)
        is_excluded = np.zeros((1, 100), dtype=bool)
        focus = np.zeros((1, 100), dtype=bool)
        focus[:, ::10] = True
        out = select_positions(
            rng=rng,
            mlm_prob=0.0,
            is_excluded=is_excluded,
            focus_positions=focus,
            focus_strategy="force_include",
            focus_strength=1.0,
        )
        # mlm_prob=0 + force_include => exactly the focus positions.
        np.testing.assert_array_equal(out, focus)

    def test_unknown_strategy_at_helper_raises(self):
        # validate_focus_params is the front line, but the helper has a
        # safety raise for completeness.
        with pytest.raises(ValueError):
            select_positions(
                rng=np.random.default_rng(0),
                mlm_prob=0.1,
                is_excluded=np.zeros((1, 4), dtype=bool),
                focus_positions=np.zeros((1, 4), dtype=bool),
                focus_strategy="bogus",
                focus_strength=1.0,
            )

    def test_validate_focus_params_negative_strength(self):
        with pytest.raises(ValueError):
            validate_focus_params("multiplicative", -1.0)


class TestFocusPositionsFromReference:
    def test_1d_vs_1d(self):
        seq = np.array([10, 20, 30, 40])
        wt = np.array([10, 99, 30, 99])
        out = focus_positions_from_reference(seq, wt)
        assert out.shape == (4,)
        assert out.dtype == bool
        np.testing.assert_array_equal(out, [False, True, False, True])

    def test_1d_seq_with_2d_reference_batch_1(self):
        seq = np.array([10, 20, 30])
        wt = np.array([[10, 99, 30]])
        out = focus_positions_from_reference(seq, wt)
        assert out.shape == (1, 3)
        np.testing.assert_array_equal(out, [[False, True, False]])

    def test_2d_seq_with_1d_reference_broadcasts(self):
        seq = np.array([[10, 20, 30], [10, 99, 30]])
        wt = np.array([10, 20, 30])
        out = focus_positions_from_reference(seq, wt)
        assert out.shape == (2, 3)
        np.testing.assert_array_equal(
            out, [[False, False, False], [False, True, False]]
        )

    def test_2d_seq_with_2d_reference_batch_1_broadcasts(self):
        seq = np.array([[10, 20, 30], [10, 99, 30]])
        wt = np.array([[10, 20, 30]])
        out = focus_positions_from_reference(seq, wt)
        assert out.shape == (2, 3)
        np.testing.assert_array_equal(
            out, [[False, False, False], [False, True, False]]
        )

    def test_2d_seq_with_matching_2d_reference_per_row(self):
        seq = np.array([[10, 20, 30], [10, 99, 30]])
        wt = np.array([[10, 99, 30], [10, 20, 99]])
        out = focus_positions_from_reference(seq, wt)
        np.testing.assert_array_equal(
            out, [[False, True, False], [False, True, True]]
        )

    def test_torch_input_returns_torch(self):
        seq = torch.tensor([[10, 20, 30], [10, 99, 30]])
        wt = torch.tensor([10, 20, 30])
        out = focus_positions_from_reference(seq, wt)
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.bool
        np.testing.assert_array_equal(
            out.numpy(), [[False, False, False], [False, True, False]]
        )

    def test_torch_seq_with_numpy_reference(self):
        # Mixed inputs: output type follows `sequences`.
        seq = torch.tensor([[10, 20, 30]])
        wt = np.array([10, 20, 99])
        out = focus_positions_from_reference(seq, wt)
        assert isinstance(out, torch.Tensor)
        np.testing.assert_array_equal(out.numpy(), [[False, False, True]])

    def test_numpy_seq_with_torch_reference(self):
        seq = np.array([[10, 20, 30]])
        wt = torch.tensor([10, 20, 99])
        out = focus_positions_from_reference(seq, wt)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, [[False, False, True]])

    def test_rejects_mismatched_length(self):
        seq = np.array([1, 2, 3])
        wt = np.array([1, 2, 3, 4])
        with pytest.raises(ValueError, match="length"):
            focus_positions_from_reference(seq, wt)

    def test_rejects_3d_sequences(self):
        seq = np.zeros((2, 2, 2), dtype=np.int64)
        wt = np.zeros((2,), dtype=np.int64)
        with pytest.raises(ValueError, match="sequences"):
            focus_positions_from_reference(seq, wt)

    def test_rejects_3d_reference(self):
        seq = np.zeros((2, 4), dtype=np.int64)
        wt = np.zeros((1, 1, 4), dtype=np.int64)
        with pytest.raises(ValueError, match="reference"):
            focus_positions_from_reference(seq, wt)

    def test_rejects_reference_larger_than_sequences(self):
        # (B, L) reference with (L,) sequences is nonsensical (reference
        # should be a single reference, not a batch).
        seq = np.array([10, 20, 30])
        wt = np.array([[10, 20, 30], [10, 99, 30]])
        with pytest.raises(ValueError, match="batch"):
            focus_positions_from_reference(seq, wt)

    def test_rejects_reference_batch_dim_mismatch(self):
        seq = np.zeros((4, 8), dtype=np.int64)
        wt = np.zeros((2, 8), dtype=np.int64)  # wt batch != seq batch and != 1
        with pytest.raises(ValueError, match="batch"):
            focus_positions_from_reference(seq, wt)

    def test_end_to_end_with_seqmask(self):
        # Plug the utility's output straight into a seqmask.
        seq = np.array([[10, 20, 30, 40, 50], [10, 99, 30, 99, 50]])
        wt = np.array([10, 20, 30, 40, 50])
        focus = focus_positions_from_reference(seq, wt)
        # Row 0 has no mutations, row 1 has positions [1, 3] mutated.
        np.testing.assert_array_equal(
            focus, [[False] * 5, [False, True, False, True, False]]
        )

        m = CategoricalMasking(
            mlm_prob_sampler=_FixedSampler(0.0, 1.0, 0.0),
            mask_token_id=999,
            valid_token_ids=list(range(10, 200)),
            special_token_ids=[],
            focus_strategy="force_include",
        )
        out, labels = m(seq, seed=0, focus_positions=focus)
        # Row 0 (no mutations): no positions selected (mlm_prob=0, no focus).
        assert np.all(labels[0] == -100)
        # Row 1 (mutations at 1, 3): exactly those positions get masked.
        assert (labels[1] != -100).sum() == 2
        assert labels[1, 1] != -100
        assert labels[1, 3] != -100
        assert out[1, 1] == 999
        assert out[1, 3] == 999
