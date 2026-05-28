import os

os.environ[
    "SEQMASK_BACKEND"
] = "torch"  # force PyTorch for deterministic testing

import numpy as np
import pytest
import torch

from seqmask.utils import create_default_mlm_weights
from seqmask.utils import ensure_non_negative
from seqmask.utils import seed_from_components


class TestEnsureNonNegative:
    def test_accepts_list_of_non_negatives(self):
        # Returns the original input unchanged.
        values = [0.0, 0.1, 1.5]
        assert ensure_non_negative(values) is values

    def test_accepts_empty_iterable(self):
        ensure_non_negative([])

    def test_accepts_scalar_zero(self):
        ensure_non_negative(0.0)

    def test_accepts_scalar_positive(self):
        ensure_non_negative(1.5)

    def test_rejects_scalar_negative(self):
        with pytest.raises(ValueError):
            ensure_non_negative(-0.0001)

    def test_rejects_single_negative_in_list(self):
        with pytest.raises(ValueError):
            ensure_non_negative([0.1, -0.5, 0.7])

    def test_rejects_all_negative_list(self):
        with pytest.raises(ValueError):
            ensure_non_negative([-1.0, -2.0])

    def test_accepts_numpy_array(self):
        ensure_non_negative(np.array([0.0, 1.0, 2.0]))

    def test_rejects_numpy_array_with_negative(self):
        with pytest.raises(ValueError):
            ensure_non_negative(np.array([0.0, -1.0]))

    def test_accepts_torch_tensor(self):
        ensure_non_negative(torch.tensor([0.0, 1.0]))

    def test_rejects_torch_tensor_with_negative(self):
        with pytest.raises(ValueError):
            ensure_non_negative(torch.tensor([0.0, -1.0]))

    def test_error_message_includes_name(self):
        with pytest.raises(ValueError, match="my_weights"):
            ensure_non_negative([-1.0], name="my_weights")


class TestSeedFromComponents:
    def test_returns_int(self):
        assert isinstance(seed_from_components(1, 2, 3), int)

    def test_deterministic_for_same_input(self):
        assert seed_from_components(1, 2, 3) == seed_from_components(1, 2, 3)

    def test_different_inputs_give_different_seeds(self):
        a = seed_from_components(1, 2, 3)
        b = seed_from_components(1, 2, 4)
        assert a != b

    def test_order_matters(self):
        assert seed_from_components(1, 2, 3) != seed_from_components(3, 2, 1)

    def test_accepts_single_component(self):
        assert isinstance(seed_from_components(42), int)

    def test_accepts_many_components(self):
        assert isinstance(seed_from_components(*range(10)), int)

    def test_fits_in_unsigned_64_bit_range(self):
        # blake2b digest_size=8 → 64-bit, big-endian. Must be in [0, 2^64).
        seed = seed_from_components(123, 456, 789)
        assert 0 <= seed < 2**64

    def test_can_seed_numpy_rng(self):
        seed = seed_from_components(1, 2, 3)
        # Must not raise; must be usable as a seed.
        rng = np.random.default_rng(seed)
        _ = rng.random(4)

    def test_zero_components_returns_int(self):
        # Currently allowed by the implementation — joining an empty tuple is "".
        assert isinstance(seed_from_components(), int)


class TestCreateDefaultMlmWeights:
    def test_returns_numpy_array(self):
        weights = create_default_mlm_weights(0.0, 1.0, 0.1)
        assert isinstance(weights, np.ndarray)

    def test_endpoints_match(self):
        weights = create_default_mlm_weights(0.0, 1.0, 0.1)
        assert weights[0] == pytest.approx(0.0)
        assert weights[-1] == pytest.approx(1.0)

    def test_number_of_steps(self):
        # 0.0 to 1.0 with 0.1 increment → 11 points (0, 0.1, ..., 1.0).
        weights = create_default_mlm_weights(0.0, 1.0, 0.1)
        assert len(weights) == 11

    def test_values_evenly_spaced(self):
        weights = create_default_mlm_weights(0.0, 1.0, 0.25)
        np.testing.assert_allclose(weights, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_single_point_when_lower_equals_upper(self):
        weights = create_default_mlm_weights(0.5, 0.5, 0.1)
        assert len(weights) == 1
        assert weights[0] == pytest.approx(0.5)

    def test_rejects_negative_lower(self):
        with pytest.raises(ValueError):
            create_default_mlm_weights(-1.0, 1.0, 0.5)

    def test_rejects_negative_upper(self):
        # `upper` is also validated via ensure_non_negative.
        with pytest.raises(ValueError):
            create_default_mlm_weights(0.0, -0.5, 0.1)

    def test_increment_rounded(self):
        # 0.0 to 1.0 with 0.3 increment → round((1.0/0.3)) + 1 = 4 points
        # spread evenly from 0 to 1, not at exact 0.3 multiples.
        weights = create_default_mlm_weights(0.0, 1.0, 0.3)
        assert len(weights) == 4
        assert weights[0] == pytest.approx(0.0)
        assert weights[-1] == pytest.approx(1.0)
