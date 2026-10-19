import csv
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np

from rl_navigation import ACTIONS, Observation, QPolicy, Demonstrations, execute_logged, run_session, learn_correction, write_frame
from vision_navigation import Frame


class ExecutionTests(unittest.TestCase):
    def test_manual_correction_uses_before_image_and_updates_model(self):
        dataset = self.path / 'dataset'
        dataset.mkdir()
        write_frame(dataset / 'old.jpg', self.frame.image)
        (dataset / 'labels.csv').write_text('filename,label\nold.jpg,forward\n', encoding='utf-8')
        old = Demonstrations(dataset)
        write_frame(self.path / 'manual_before.jpg', self.frame.image)
        updated = learn_correction(dataset, self.path / 'model.json', self.path,
                                   'manual', 'backward', self.frame, old)
        self.assertEqual(updated.counts, {'forward': 1, 'backward': 1})
        self.assertEqual(updated.filenames[0], 'old.jpg')
        self.assertNotEqual(updated.fingerprint, old.fingerprint)
        model = json.loads((self.path / 'model.json').read_text(encoding='utf-8'))
        self.assertEqual(model['counts']['backward'], 1)
        again = learn_correction(dataset, self.path / 'model.json', self.path,
                                 'manual', 'backward', self.frame, updated)
        self.assertEqual(len(again.filenames), 2)

    def test_start_key_during_capture_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points = root / 'points.json'
            from phone_direction_control import default_points
            points.write_text(json.dumps(default_points()))
            args = SimpleNamespace(points=points, logs=root, max_age=3, stable_frames=1,
                                   mode='drive', steps=3, seconds=30, margin=0.12, epsilon=0,
                                   allow_backward=False, pulse_ms=100, settle=0, interval_ms=100)
            demonstrations = Mock()
            demonstrations.observe.return_value = Observation('0', 'forward', True, 'ok', 0, 1)
            policy = Mock(updates=0)
            policy.choose.return_value = 'forward'
            workers = []
            keys = iter([-1, ord('G'), -1, ord('q')])

            def thread_factory(target, **kwargs):
                return SimpleNamespace(start=lambda: workers.append(target))

            def wait_key(delay):
                key = next(keys)
                if key == ord('G') or key == ord('q'):
                    while workers:
                        workers.pop(0)()
                return key

            def fresh_frame(*args):
                now = time.monotonic()
                return Frame(np.zeros((48, 96, 3), dtype=np.uint8), 2800, 1260, now, now)

            with patch('rl_navigation.threading.Thread', side_effect=thread_factory), \
                 patch('rl_navigation.capture', side_effect=fresh_frame), \
                 patch('rl_navigation.execute_logged', side_effect=lambda *args: fresh_frame()) as execute, \
                 patch('rl_navigation.cv2.namedWindow'), patch('rl_navigation.cv2.imshow'), \
                 patch('rl_navigation.cv2.waitKey', side_effect=wait_key), \
                 patch('rl_navigation.cv2.getWindowProperty', return_value=1), \
                 patch('rl_navigation.cv2.destroyAllWindows'):
                run_session(args, demonstrations, policy, Mock())
            execute.assert_called_once()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        now = time.monotonic()
        self.frame = Frame(np.zeros((48, 96, 3), dtype=np.uint8), 2800, 1260, now, now)
        self.observation = Observation('0:stop', 'forward', True, 'ok', 0.1, 0.9)

    def test_reverse_pairs_command_and_images(self):
        controller = Mock()
        after = Frame(self.frame.image, 2800, 1260, self.frame.started + 0.3, self.frame.started + 0.4)
        with patch('rl_navigation.capture', return_value=after), patch('rl_navigation.time.sleep'):
            result = execute_logged(controller, (100, 200), 'backward', self.frame,
                                    self.observation, self.path, 'step', 100, 0, 100, 'manual')
        controller.press.assert_called_once_with((100, 200), 100)
        self.assertIs(result, after)
        metadata = json.loads((self.path / 'step.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['button'], 'down')
        self.assertEqual(metadata['status'], 'completed')
        self.assertAlmostEqual(metadata['actual_capture_interval_ms'], 300, places=3)
        self.assertTrue((self.path / metadata['before_image']).exists())
        self.assertTrue((self.path / metadata['after_image']).exists())
        with (self.path / 'review.csv').open(encoding='utf-8-sig', newline='') as stream:
            self.assertEqual(next(csv.DictReader(stream))['action'], 'backward')

    def test_failed_touch_has_durable_error_and_before_image(self):
        controller = Mock()
        controller.press.side_effect = RuntimeError('disconnected')
        with self.assertRaises(RuntimeError):
            execute_logged(controller, (1, 2), 'forward', self.frame, self.observation,
                           self.path, 'failed', 100, 0, 100, 'policy')
        metadata = json.loads((self.path / 'failed.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['status'], 'error')
        self.assertTrue((self.path / metadata['before_image']).exists())
        self.assertFalse((self.path / metadata['after_image']).exists())

    def test_logging_failure_prevents_touch(self):
        controller = Mock()
        with patch('rl_navigation.write_frame', side_effect=OSError('full disk')):
            with self.assertRaises(OSError):
                execute_logged(controller, (1, 2), 'forward', self.frame, self.observation,
                               self.path, 'full', 100, 0, 100, 'policy')
        controller.press.assert_not_called()

    def test_legacy_q_values_keep_stop_and_append_backward(self):
        path = self.path / 'policy.json'
        path.write_text(json.dumps({'fingerprint': 'abc', 'actions': list(ACTIONS[:-1]),
                                    'q_values': {'0:stop': [1, 2, 3, 4]}, 'updates': 7}))
        policy = QPolicy(path, 'abc')
        self.assertEqual(policy.values['0:stop'], [1, 2, 3, 4, 0])
        self.assertEqual(policy.updates, 7)

    def test_backward_is_not_automatically_explored_unless_enabled(self):
        policy = QPolicy(self.path / 'policy.json', 'abc')
        policy.row(self.observation)[ACTIONS.index('backward')] = 20
        self.assertNotEqual(policy.choose(self.observation), 'backward')
        self.assertEqual(policy.choose(self.observation, allow_backward=True), 'backward')
        for _ in range(30):
            self.assertNotEqual(policy.choose(self.observation, epsilon=1), 'backward')


if __name__ == '__main__':
    unittest.main()
