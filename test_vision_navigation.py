import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import time

import cv2
import numpy as np

from vision_navigation import DecisionGate, Frame, analyze, monitor, propose, validate_drive


def make_profile():
    color = cv2.cvtColor(np.uint8([[[100, 150, 180]]]), cv2.COLOR_BGR2LAB)[0, 0].tolist()
    return {
        'screen_size': [2800, 1260], 'road_roi': [0, 10, 300, 180],
        'exclude_roi': [], 'floor_lab': color, 'color_tolerance': 28,
        'anchor_roi': [0, 0, 20, 10], 'anchor_gray': np.zeros((10, 20), dtype=int).tolist(),
        'anchor_error_limit': 16, 'clear_ratio': 0.88, 'near_clear_ratio': 0.95,
        'max_frame_age_s': 1.5, 'freeze_s': 3,
        'controls_verified': False, 'release_stops_verified': False,
        'actions': {'forward': 'up', 'turn_left': 'up_left', 'turn_right': 'up_right'},
    }


def make_image():
    image = np.zeros((190, 300, 3), dtype=np.uint8)
    image[10:] = (100, 150, 180)
    return image


class VisionTests(unittest.TestCase):
    def test_clear_floor_forward(self):
        profile = make_profile()
        _, _, _, scores, near, _ = analyze(make_image(), profile)
        self.assertEqual(propose(scores, near, profile)[0], 'forward')

    def test_close_obstacle_stops(self):
        profile = make_profile()
        image = make_image()
        image[145:175, 110:190] = (0, 0, 255)
        _, _, _, scores, near, _ = analyze(image, profile)
        self.assertEqual(propose(scores, near, profile)[0], 'stop')

    def test_far_obstacle_and_clear_left(self):
        profile = make_profile()
        image = make_image()
        image[10:100, 100:] = (0, 0, 255)
        _, _, _, scores, near, _ = analyze(image, profile)
        self.assertEqual(propose(scores, near, profile)[0], 'turn_left')

    def test_disconnected_floor_not_passable(self):
        profile = make_profile()
        image = make_image()
        image[125:145, :] = (0, 0, 255)
        _, floor, _, _, _, _ = analyze(image, profile)
        self.assertFalse(floor[:100].any())

    def test_heavily_masked_corridor_unknown(self):
        profile = make_profile()
        profile['exclude_roi'] = [[100, 0, 100, 180]]
        _, _, _, scores, near, _ = analyze(make_image(), profile)
        self.assertEqual(scores['center'], 0)
        self.assertEqual(propose(scores, near, profile)[0], 'stop')

    def test_equal_turns_stop(self):
        self.assertEqual(propose({'left': 1, 'center': 0.5, 'right': 1},
                                 {'left': 1, 'center': 1, 'right': 1}, make_profile())[0], 'stop')

    def test_three_consistent_frames(self):
        gate = DecisionGate()
        image = make_image()
        frame = Frame(image, 2800, 1260, 10, 10.1)
        results = [gate.evaluate(frame, image, 'forward', 0, make_profile(), now)[0]
                   for now in (10.1, 10.2, 10.3)]
        self.assertEqual(results, ['wait', 'wait', 'forward'])

    def test_stale_frame_resets_consensus(self):
        gate = DecisionGate()
        image = make_image()
        profile = make_profile()
        gate.stable = 10
        gate.candidate = 'forward'
        decision = gate.evaluate(Frame(image, 2800, 1260, 10, 10.1), image, 'forward', 0, profile, 12)
        self.assertEqual(decision[0], 'stop')
        self.assertEqual(gate.stable, 0)

    def test_anchor_change_stops(self):
        image = make_image()
        result = DecisionGate().evaluate(Frame(image, 2800, 1260, 10, 10.1), image,
                                         'forward', 50, make_profile(), 10.2)
        self.assertEqual(result[0], 'stop')

    def test_freeze_history_survives_action(self):
        gate = DecisionGate()
        image = make_image()
        profile = make_profile()
        gate.evaluate(Frame(image, 2800, 1260, 10, 10.1), image, 'forward', 0, profile, 10.2)
        gate.reset_consensus()
        decision = gate.evaluate(Frame(image, 2800, 1260, 14, 14.1), image, 'forward', 0, profile, 14.2)
        self.assertIn('frozen', decision[1])

    def test_drive_requires_verified_release(self):
        with tempfile.TemporaryDirectory() as directory:
            points = Path(directory) / 'points.json'
            points.write_text('{}')
            profile = make_profile()
            profile['controls_verified'] = True
            with self.assertRaises(ValueError):
                validate_drive(profile, points)
            profile['release_stops_verified'] = True
            validate_drive(profile, points)

    def test_observe_mode_never_sends_touch_even_with_g(self):
        image = make_image()
        profile = make_profile()
        profile.update(max_steps=20, max_turns=3, pulse_ms=100, settle_s=0.5)
        now = time.monotonic()
        frame = Frame(image, 2800, 1260, now, now)
        source = Mock()
        source.messages.get_nowait.side_effect = [frame, frame, frame]
        controller = Mock()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with patch('vision_navigation.FrameSource', return_value=source), \
                 patch('vision_navigation.cv2.namedWindow'), \
                 patch('vision_navigation.cv2.imshow'), \
                 patch('vision_navigation.cv2.destroyAllWindows'), \
                 patch('vision_navigation.cv2.getWindowProperty', return_value=1), \
                 patch('vision_navigation.cv2.waitKey', side_effect=[ord('g'), ord('g'), ord('q')]):
                monitor(controller, profile, base / 'missing-points.json', False, base / 'log.jsonl')
            controller.press.assert_not_called()


if __name__ == '__main__':
    unittest.main()
