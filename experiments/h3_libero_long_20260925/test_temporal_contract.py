"""These tests need no Torch, dataset, accelerator, or simulator."""
import unittest

from temporal_contract import Clock, causal_window, valid_anchors


def window(length, anchor):
    return causal_window(length, anchor, observation_timing="post_action")


class TemporalTests(unittest.TestCase):
    def test_initial_observation_never_targets_already_executed_action(self):
        value = window(80, 0)
        self.assertEqual(value.future_actions, tuple(range(1, 9)))
        self.assertEqual(value.lower_observations, (0,) * 8)
        self.assertEqual(value.upper_observations, (0,) * 16)
        self.assertEqual(value.upper_actions, (None,) * 16)
        self.assertEqual(value.upper_action_mask, 0)

    def test_second_chunk_uses_exactly_the_same_upper_context(self):
        one, two = window(80, 16), window(80, 24)
        for key in ("boundary", "upper_observations", "upper_actions", "upper_teacher"):
            self.assertEqual(getattr(one, key), getattr(two, key))
        self.assertEqual(two.lower_observations[-1], 24)
        self.assertEqual(two.upper_observations[-1], 16)
        self.assertEqual(two.future_actions[0], 25)

    def test_past_actions_include_the_one_that_produced_boundary_observation(self):
        value = window(80, 16)
        self.assertEqual(value.upper_actions, tuple(range(1, 17)))
        self.assertEqual(value.upper_action_mask, 1)
        self.assertEqual(value.upper_teacher, 32)

    def test_episode_end_does_not_enter_remaining_condition(self):
        near, far = window(41, 32), window(90, 32)
        self.assertEqual(near.remaining_condition, far.remaining_condition)
        self.assertEqual(near.remaining_condition, 1.0)
        self.assertEqual(near.upper_teacher, 40)
        self.assertEqual(far.upper_teacher, 48)

    def test_no_fake_future_action_padding(self):
        self.assertEqual(list(valid_anchors(8)), [])
        self.assertEqual(list(valid_anchors(9)), [0])
        self.assertEqual(list(valid_anchors(17)), [0, 8])
        with self.assertRaises(ValueError):
            window(16, 8)

    def test_exhaustive_small_episode_boundaries(self):
        for length in range(1, 200):
            for anchor in valid_anchors(length):
                value = window(length, anchor)
                self.assertTrue(all(0 <= i <= anchor for i in value.lower_observations))
                self.assertTrue(all(0 <= i <= value.boundary for i in value.upper_observations))
                self.assertTrue(all(i is None or 0 <= i <= value.boundary for i in value.upper_actions))
                self.assertTrue(all(anchor < i < length for i in value.future_actions))
                self.assertLess(value.boundary, value.upper_teacher)
                self.assertLess(value.upper_teacher, length)

    def test_missing_timing_audit_is_rejected(self):
        for timing in (None, "pre_action", "unknown"):
            with self.assertRaises(ValueError):
                causal_window(40, 0, observation_timing=timing)

    def test_invalid_indices_are_rejected(self):
        for length, anchor in ((40, -8), (40, 1), (40, 40), (True, 0), (40, False), (40., 0)):
            with self.assertRaises(ValueError):
                window(length, anchor)

    def test_clock_holds_upper_for_two_chunks(self):
        clock = Clock()
        with self.assertRaises(ValueError):
            clock.record_chunk()
        for boundary in (0, 16, 32):
            self.assertEqual(clock.step, boundary)
            self.assertTrue(clock.needs_upper())
            clock.record_upper()
            clock.record_chunk()
            self.assertFalse(clock.needs_upper())
            with self.assertRaises(ValueError):
                clock.record_upper()
            clock.record_chunk()
        self.assertEqual(clock.step, 48)


if __name__ == "__main__":
    unittest.main()
