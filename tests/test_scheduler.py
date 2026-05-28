import os

os.environ[
    "SEQMASK_BACKEND"
] = "torch"  # force PyTorch for deterministic testing

import pytest

from seqmask.scheduler import EMAScheduler
from seqmask.scheduler import LinearScheduler
from seqmask.scheduler import Scheduler
from seqmask.scheduler import SharedCounter
from seqmask.scheduler import StagedLinearScheduler


class TestSharedCounter:
    def test_init_starts_at_initial_value(self):
        c = SharedCounter(initial_value=3, final_value=10)
        assert c.get() == 3
        assert c.value == 3

    def test_step_increments_by_one(self):
        c = SharedCounter(initial_value=0, final_value=10)
        c.step()
        assert c.value == 1
        c.step()
        assert c.value == 2

    def test_step_caps_at_final_value(self):
        c = SharedCounter(initial_value=0, final_value=2)
        for _ in range(10):
            c.step()
        assert c.value == 2

    def test_reset_returns_to_initial_value(self):
        c = SharedCounter(initial_value=5, final_value=10)
        c.step()
        c.step()
        c.reset()
        assert c.value == 5

    def test_reset_from_zero_initial(self):
        c = SharedCounter(initial_value=0, final_value=10)
        c.step()
        c.step()
        c.reset()
        assert c.value == 0

    def test_value_property_matches_get(self):
        c = SharedCounter(initial_value=0, final_value=5)
        c.step()
        assert c.value == c.get()

    def test_initial_value_equals_final_value_does_not_step(self):
        c = SharedCounter(initial_value=5, final_value=5)
        c.step()
        assert c.value == 5


class TestSchedulerBase:
    def test_step_and_reset_delegate_to_counter(self):
        s = Scheduler(initial_value=0, final_value=10)
        s.step()
        assert s.current_step.value == 1
        s.reset()
        assert s.current_step.value == 0

    def test_sample_is_not_implemented(self):
        s = Scheduler(initial_value=0, final_value=10)
        with pytest.raises(NotImplementedError):
            s.sample()


class TestLinearScheduler:
    def test_rejects_mismatched_length_weights(self):
        with pytest.raises(ValueError):
            LinearScheduler(
                initial_weights=[0.1, 0.2],
                final_weights=[1.0, 1.0, 1.0],
                total_steps=10,
            )

    @pytest.mark.parametrize("bad_total", [0, -1])
    def test_rejects_non_positive_total_steps(self, bad_total):
        with pytest.raises(ValueError):
            LinearScheduler(
                initial_weights=[0.0],
                final_weights=[1.0],
                total_steps=bad_total,
            )

    def test_rejects_negative_initial_weights(self):
        with pytest.raises(ValueError):
            LinearScheduler(
                initial_weights=[-0.1, 0.5],
                final_weights=[1.0, 1.0],
                total_steps=10,
            )

    def test_rejects_negative_final_weights(self):
        with pytest.raises(ValueError):
            LinearScheduler(
                initial_weights=[0.1, 0.5],
                final_weights=[1.0, -1.0],
                total_steps=10,
            )

    def test_sample_at_step_zero_returns_initial(self):
        sched = LinearScheduler(
            initial_weights=[0.1, 0.2, 0.3],
            final_weights=[1.0, 1.0, 1.0],
            total_steps=10,
        )
        assert sched.sample() == [0.1, 0.2, 0.3]

    def test_sample_after_total_steps_returns_final(self):
        sched = LinearScheduler(
            initial_weights=[0.0, 0.5],
            final_weights=[1.0, 1.0],
            total_steps=4,
        )
        for _ in range(4):
            sched.step()
        assert sched.sample() == pytest.approx([1.0, 1.0])

    def test_sample_clamped_past_total_steps(self):
        # Progress must clamp to 1.0; weights stay at final.
        sched = LinearScheduler(
            initial_weights=[0.0],
            final_weights=[1.0],
            total_steps=4,
        )
        for _ in range(100):
            sched.step()
        # current_step itself is also capped at 4 by SharedCounter.
        assert sched.current_step.value == 4
        assert sched.sample() == pytest.approx([1.0])

    def test_sample_mid_step(self):
        sched = LinearScheduler(
            initial_weights=[0.0, 0.0],
            final_weights=[1.0, 2.0],
            total_steps=10,
        )
        for _ in range(5):
            sched.step()
        # progress = 0.5 → weights = [0.5, 1.0]
        assert sched.sample() == pytest.approx([0.5, 1.0])

    def test_monotonic_increase_for_increasing_targets(self):
        sched = LinearScheduler(
            initial_weights=[0.0],
            final_weights=[1.0],
            total_steps=10,
        )
        prev = sched.sample()[0]
        for _ in range(10):
            sched.step()
            cur = sched.sample()[0]
            assert cur >= prev
            prev = cur

    def test_reset_returns_to_initial(self):
        sched = LinearScheduler(
            initial_weights=[0.1, 0.2],
            final_weights=[1.0, 1.0],
            total_steps=10,
        )
        for _ in range(5):
            sched.step()
        sched.reset()
        assert sched.sample() == [0.1, 0.2]


class TestStagedLinearScheduler:
    @pytest.mark.parametrize("bad_n", [0, -1])
    def test_rejects_non_positive_num_stages(self, bad_n):
        with pytest.raises(ValueError):
            StagedLinearScheduler(num_stages=bad_n, total_steps=10)

    @pytest.mark.parametrize("bad_total", [0, -1])
    def test_rejects_non_positive_total_steps(self, bad_total):
        with pytest.raises(ValueError):
            StagedLinearScheduler(num_stages=3, total_steps=bad_total)

    def test_rejects_total_steps_below_num_stages(self):
        with pytest.raises(ValueError):
            StagedLinearScheduler(num_stages=3, total_steps=2)

    def test_first_stage_is_always_active_at_step_zero(self):
        sched = StagedLinearScheduler(num_stages=3, total_steps=30)
        weights = sched.sample()
        # At step 0 all stages are still at 0; implementation forces
        # weights[0]=1 to avoid degenerate (all-zero) distributions.
        assert weights[0] == pytest.approx(1.0)
        assert sum(weights) == pytest.approx(1.0)

    def test_all_stages_active_at_end(self):
        sched = StagedLinearScheduler(num_stages=3, total_steps=30)
        for _ in range(30):
            sched.step()
        weights = sched.sample()
        assert weights == pytest.approx([1 / 3, 1 / 3, 1 / 3])
        assert len(weights) == sched.num_stages

    def test_weights_always_sum_to_one(self):
        sched = StagedLinearScheduler(num_stages=4, total_steps=40)
        for _ in range(40):
            assert sum(sched.sample()) == pytest.approx(1.0)
            sched.step()

    def test_weights_length_matches_num_stages(self):
        sched = StagedLinearScheduler(num_stages=5, total_steps=50)
        assert len(sched.sample()) == 5

    def test_num_stages_stored(self):
        sched = StagedLinearScheduler(num_stages=4, total_steps=40)
        assert sched.num_stages == 4
        assert sched.total_steps == 40

    def test_get_weights_matches_sample(self):
        sched = StagedLinearScheduler(num_stages=3, total_steps=30)
        for _ in range(10):
            sched.step()
        assert sched.get_weights() == sched.sample()


class TestEMAScheduler:
    def test_rejects_mismatched_weight_lengths(self):
        with pytest.raises(ValueError):
            EMAScheduler(initial_weights=[0.1], final_weights=[1.0, 1.0])

    def test_rejects_beta_outside_open_unit_interval(self):
        with pytest.raises(ValueError):
            EMAScheduler(initial_weights=[0.0], final_weights=[1.0], beta=0.0)
        with pytest.raises(ValueError):
            EMAScheduler(initial_weights=[0.0], final_weights=[1.0], beta=1.0)
        with pytest.raises(ValueError):
            EMAScheduler(initial_weights=[0.0], final_weights=[1.0], beta=-0.1)

    def test_rejects_negative_weights(self):
        with pytest.raises(ValueError):
            EMAScheduler(initial_weights=[-0.1], final_weights=[1.0])
        with pytest.raises(ValueError):
            EMAScheduler(initial_weights=[0.0], final_weights=[-1.0])

    @pytest.mark.parametrize("bad_multiplier", [0.0, -1.0])
    def test_rejects_non_positive_multiplier(self, bad_multiplier):
        with pytest.raises(ValueError):
            EMAScheduler(
                initial_weights=[0.0],
                final_weights=[1.0],
                multiplier=bad_multiplier,
            )

    def test_sample_at_step_zero_returns_initial(self):
        sched = EMAScheduler(
            initial_weights=[0.1, 0.5], final_weights=[1.0, 1.0], beta=0.9
        )
        # decay_factor = beta**0 = 1 → returns initial weights.
        assert sched.sample() == pytest.approx([0.1, 0.5])

    def test_sample_converges_toward_final(self):
        sched = EMAScheduler(
            initial_weights=[0.0],
            final_weights=[1.0],
            beta=0.5,
            multiplier=1.0,
        )
        for _ in range(50):
            sched.step()
        weight = sched.sample()[0]
        assert weight == pytest.approx(1.0, abs=1e-6)

    def test_sample_monotonic_toward_final(self):
        sched = EMAScheduler(
            initial_weights=[0.0], final_weights=[1.0], beta=0.9
        )
        prev = sched.sample()[0]
        for _ in range(50):
            sched.step()
            cur = sched.sample()[0]
            assert cur >= prev
            prev = cur

    def test_multiplier_speeds_up_decay(self):
        slow = EMAScheduler(
            initial_weights=[0.0],
            final_weights=[1.0],
            beta=0.5,
            multiplier=1.0,
        )
        fast = EMAScheduler(
            initial_weights=[0.0],
            final_weights=[1.0],
            beta=0.5,
            multiplier=4.0,
        )
        for _ in range(3):
            slow.step()
            fast.step()
        # With the same beta, a larger multiplier should be closer to final.
        assert fast.sample()[0] > slow.sample()[0]

    def test_explicit_formula_matches(self):
        beta = 0.9
        sched = EMAScheduler(
            initial_weights=[0.0, 2.0],
            final_weights=[1.0, 0.0],
            beta=beta,
            multiplier=1.0,
        )
        for _ in range(5):
            sched.step()
        decay = beta**5
        expected = [
            decay * 0.0 + (1 - decay) * 1.0,
            decay * 2.0 + (1 - decay) * 0.0,
        ]
        assert sched.sample() == pytest.approx(expected)
