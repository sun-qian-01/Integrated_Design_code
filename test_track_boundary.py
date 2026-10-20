import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from track_boundary import TrackBoundary, color_features, polygon_mask, train


class TrackTests(unittest.TestCase):
    def model(self):
        image = np.array([[(80, 55, 40), (100, 110, 100)]], np.uint8)
        colors = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[0].astype(float)
        return TrackBoundary({'version': 1, 'space': 'blurred-Lab',
                              'centers': {'road': [colors[0].tolist()], 'background': [colors[1].tolist()]},
                              'radius': {'road': 15, 'background': 15}})

    def test_unknown_colors_are_not_assumed_road(self):
        model = self.model()
        pixels = np.array([model.centers['road'][0], model.centers['background'][0], [255, 0, 255]], np.float32)
        self.assertEqual(model.classify(pixels).tolist(), [1, 0, -1])

    def test_normalized_polygons(self):
        mask = polygon_mask((20, 40, 3), [[[0, 0], [0.5, 0], [0.5, 1], [0, 1]]])
        self.assertEqual(mask[10, 10], 255)
        self.assertEqual(mask[10, 35], 0)

    def test_out_of_track_stops_without_touch(self):
        model = self.model()
        image = np.full((432, 960, 3), (100, 110, 100), np.uint8)
        action, duration, details = model.constrain(image, 'up_right', 500, 150)
        self.assertEqual(action, 'stop')
        self.assertFalse(details['usable'])

    def test_policy_stop_preserved_and_turns_shortened(self):
        model = self.model()
        result = {'usable': True, 'suggestion': 'up_left', 'corridors': {'up_right': {'background': 0.1}}}
        with patch.object(model, 'analyze', return_value=result):
            self.assertEqual(model.constrain(None, 'stop', 500, 150)[0], 'stop')
            action, duration, details = model.constrain(None, 'up_right', 500, 150)
            self.assertEqual((action, duration), ('up_right', 150))
            self.assertEqual(details['proposed'], 'up_right')
            self.assertEqual(model.constrain(None, 'forward', 500, 150)[1], 250)
            self.assertEqual(model.constrain(None, 'up_right', 100, 150)[1], 100)

    def test_censored_center_does_not_override_policy_without_background_evidence(self):
        model = self.model()
        result = {'usable': True, 'suggestion': 'up_left', 'corridors': {'forward': {'background': 0.1}}}
        with patch.object(model, 'analyze', return_value=result):
            self.assertEqual(model.constrain(None, 'forward', 500, 150)[0], 'forward')

    def test_background_collision_redirects_only_to_visible_road(self):
        model = self.model()
        result = {'usable': True, 'suggestion': 'up_left', 'corridors': {
            'forward': {'background': 0.8, 'road': 0.1}, 'up_left': {'background': 0.05, 'road': 0.8}}}
        with patch.object(model, 'analyze', return_value=result):
            self.assertEqual(model.constrain(None, 'forward', 500, 150)[:2], ('up_left', 150))
            result['corridors']['up_left']['road'] = 0.1
            self.assertEqual(model.constrain(None, 'forward', 500, 150)[0], 'stop')

    def test_train_uses_explicit_masks_and_excludes_validation_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.zeros((40, 80, 3), np.uint8)
            image[:, :40] = (80, 55, 40)
            image[:, 40:] = (100, 110, 100)
            cv2.imencode('.png', image)[1].tofile(root / 'sample.png')
            cv2.imencode('.png', image)[1].tofile(root / 'held.png')
            record = {'filename': 'sample.png', 'road': [[[0, 0], [.4, 0], [.4, 1], [0, 1]]],
                      'background': [[[.6, 0], [1, 0], [1, 1], [.6, 1]]]}
            annotations = root / 'annotations.json'
            annotations.write_text(json.dumps({'images': [record, {**record, 'filename': 'held.png', 'split': 'validation'}]}), encoding='utf-8')
            with patch('builtins.print'):
                result = train(root, annotations, root / 'model.json')
            self.assertEqual(result['validation_polygons']['road']['road'], 1)
            self.assertEqual(result['validation_polygons']['background']['background'], 1)
            model = TrackBoundary.load(root / 'model.json')
            self.assertEqual(model.classify(color_features(image)[20, [10, 65]]).tolist(), [1, 0])


if __name__ == '__main__':
    unittest.main()
