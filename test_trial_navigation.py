import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from phone_direction_control import default_points
from rl_navigation import ACTIONS, Observation
from train_navigation_model import features
from trial_navigation import Trial, TrialCycle, TrialPolicy, run_trial
from vision_navigation import Frame


class TrialTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.image = np.zeros((48, 96, 3), dtype=np.uint8)
        self.demonstrations = Mock(vectors=np.array([features(self.image)]))
        self.observation = Observation('0:stop', 'forward', False, 'ambiguous labels', 15, 0.03)
        self.demonstrations.observe.return_value = self.observation
        self.policy = TrialPolicy(self.root / 'policy.json', self.demonstrations)

    def frame(self):
        now = time.monotonic()
        return Frame(self.image, 2800, 1260, now, now)

    def test_ambiguity_does_not_block_but_stop_label_does(self):
        action, _ = self.policy.choose('0', self.observation, 0, False)
        self.assertEqual(action, 'forward')
        self.observation.label = 'obstacle'
        self.assertEqual(self.policy.choose('0', self.observation, 0, False)[0], 'stop')

    def test_negative_reward_changes_next_choice_and_persists(self):
        self.policy.choose('0', self.observation, 0, False)
        self.policy.update('0', 'forward', -2, '0', True, 'forward')
        restored = TrialPolicy(self.root / 'policy.json', self.demonstrations)
        self.assertEqual(restored.updates, 1)
        self.assertNotEqual(restored.choose('0', self.observation, 0, False)[0], 'forward')

    def test_correction_changes_choice_without_fake_reward(self):
        self.policy.correct('0', 'forward', 'up_right')
        self.assertEqual(self.policy.choose('0', self.observation, 0, False)[0], 'up_right')
        self.assertEqual(self.policy.updates, 0)
        self.assertEqual(self.policy.corrections, 1)

    def test_stable_anchor_survives_added_demonstrations(self):
        self.policy.save()
        changed = Mock(vectors=np.array([features(self.image), features(255 - self.image)]))
        restored = TrialPolicy(self.root / 'policy.json', changed)
        self.assertEqual(len(restored.anchors), 1)
        self.assertEqual(restored.state(self.image), '0')

    def test_waits_for_feedback_and_accepts_static_but_fresh_frame(self):
        cycle = TrialCycle(1.5)
        frame = self.frame()
        self.assertTrue(cycle.ready(frame, frame.started)[0])
        self.assertFalse(cycle.ready(frame, frame.started + 2)[0])
        trial = Trial('id', frame, self.observation, '0', 'forward')
        cycle.begin(trial, frame.started)
        cycle.complete(frame)
        self.assertFalse(cycle.ready(self.frame(), time.monotonic())[0])
        cycle.consume()
        self.assertTrue(cycle.ready(self.frame(), time.monotonic())[0])

    def run_fake_session(self, feedback, mode='trial', expected_calls=1, click_auto=False, guard=None):
        points = self.root / 'points.json'
        points.write_text(json.dumps(default_points()))
        args = SimpleNamespace(points=points, trial_policy=self.root / 'session-policy.json',
                               logs=self.root / 'logs', max_age=3, steps=10, seconds=60,
                               pulse_ms=500, settle=0, interval_ms=500, epsilon=0,
                               allow_backward=False, margin=0.12, mode=mode)
        if guard is not None:
            args.track_model = self.root / 'track.json'
            args.track_model.write_text('{}')
            args.turn_ms = 150
        original_policy = None
        if mode == 'showcase':
            self.policy.path = args.trial_policy
            self.policy.correct('0', 'forward', 'up_right')
            original_policy = args.trial_policy.read_bytes()
        workers = []
        keys = iter([-1, ord('G'), -1, -1, feedback, -1, ord('q')])

        def wait_key(delay):
            key = next(keys)
            if click_auto and key == ord('G'):
                rendered = imshow.call_args.args[1]
                callback.call_args.args[1](1 + 3, 20, rendered.shape[0] - 30, 0, None)
                key = -1
            while workers:
                workers.pop(0)()
            return key

        with patch('track_boundary.TrackBoundary.load', return_value=guard), \
             patch('trial_navigation.threading.Thread', side_effect=lambda target, **kwargs:
                   SimpleNamespace(start=lambda: workers.append(target))), \
             patch('trial_navigation.capture', side_effect=lambda controller: self.frame()), \
             patch('trial_navigation.execute_logged', side_effect=lambda *args: self.frame()) as execute, \
             patch('trial_navigation.cv2.namedWindow'), patch('trial_navigation.cv2.imshow') as imshow, \
             patch('trial_navigation.cv2.setMouseCallback') as callback, \
             patch('trial_navigation.cv2.waitKey', side_effect=wait_key), \
             patch('trial_navigation.cv2.getWindowProperty', return_value=1), \
             patch('trial_navigation.cv2.destroyAllWindows'):
            run_trial(args, self.demonstrations, Mock())
            self.assertEqual(execute.call_count, expected_calls)
            if expected_calls:
                expected = guard.constrain.return_value[0] if guard is not None else 'up_right' if mode == 'showcase' else 'forward'
                self.assertEqual(execute.call_args.args[2], expected)
                if guard is not None:
                    self.assertEqual(execute.call_args.args[7], 150)
                    self.assertEqual(execute.call_args.args[9], 150)
        if original_policy is not None:
            self.assertEqual(args.trial_policy.read_bytes(), original_policy)
        return json.loads(args.trial_policy.read_text(encoding='utf-8'))

    def test_gui_ambiguous_frame_executes_then_waits_no_reward_no_learning(self):
        stored = self.run_fake_session(-1)
        self.assertEqual(stored['updates'], 0)

    def test_gui_feedback_updates_q_after_autonomous_attempt(self):
        stored = self.run_fake_session(ord('2'))
        self.assertEqual(stored['updates'], 1)
        self.assertLess(stored['q_values']['0'][ACTIONS.index('forward')], 0)

    def test_track_override_executes_short_pulse_and_rates_executed_action(self):
        guard = Mock()
        guard.overlay.side_effect = lambda image: image
        guard.constrain.return_value = ('up_right', 150, {'suggestion': 'up_right'})
        stored = self.run_fake_session(ord('2'), guard=guard)
        self.assertLess(stored['q_values']['0'][ACTIONS.index('up_right')], 0)
        self.assertGreater(stored['q_values']['0'][ACTIONS.index('forward')], 0)

    def test_track_stop_never_dispatches_touch(self):
        guard = Mock()
        guard.overlay.side_effect = lambda image: image
        guard.constrain.return_value = ('stop', 150, {'suggestion': 'stop'})
        self.run_fake_session(-1, expected_calls=0, guard=guard)

    def test_showcase_continues_without_feedback_and_keeps_trained_policy(self):
        stored = self.run_fake_session(-1, mode='showcase', expected_calls=2)
        self.assertEqual(stored['updates'], 0)
        self.assertEqual(stored['corrections'], 1)

    def test_auto_button_switches_trial_to_continuous_showcase(self):
        self.run_fake_session(-1, expected_calls=2, click_auto=True)

    def test_showcase_stop_does_not_send_movement(self):
        self.observation.label = 'obstacle'
        self.run_fake_session(-1, mode='showcase', expected_calls=0)

    def test_read_only_choice_does_not_create_q_state(self):
        self.policy.choose('new-state', self.observation, 0, False, remember=False)
        self.assertNotIn('new-state', self.policy.values)

    def test_positive_reward_is_saved(self):
        self.policy.row('0', 'forward')
        self.policy.update('0', 'forward', 1, '0', False, 'forward')
        self.assertGreater(self.policy.values['0'][ACTIONS.index('forward')], 0.2)
        restored = TrialPolicy(self.root / 'policy.json', self.demonstrations)
        self.assertEqual(restored.values, self.policy.values)

    def test_backward_excluded_until_enabled(self):
        self.policy.correct('0', 'forward', 'backward')
        self.assertNotEqual(self.policy.choose('0', self.observation, 0, False)[0], 'backward')
        self.assertEqual(self.policy.choose('0', self.observation, 0, True)[0], 'backward')


if __name__ == '__main__':
    unittest.main()
