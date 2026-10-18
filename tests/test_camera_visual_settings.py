import unittest
from unittest.mock import patch

import numpy as np
import pybullet as p

from environment.camera import Camera, LinkCamera


class CameraVisualSettingsTests(unittest.TestCase):
    def setUp(self):
        self.rgba = np.array(
            [[[10, 100, 200, 23], [200, 250, 255, 81]],
             [[0, 1, 50, 159], [80, 120, 180, 255]]],
            dtype=np.uint8,
        )
        self.depth = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        self.seg = np.array([[3, 4], [-1, 7]], dtype=np.int32)

    @staticmethod
    def _cameras():
        return (
            Camera((1, -2, 3), (0, 0, 0), 0.1, 6, (2, 2), 55),
            LinkCamera(0.1, 6, (2, 2), 55),
        )

    def _capture(self, camera, **kwargs):
        if isinstance(camera, LinkCamera):
            return camera.get_cam_img(17, 4, **kwargs)
        return camera.get_cam_img(**kwargs)

    def _mock_capture(self):
        return patch(
            "environment.camera.p.getCameraImage",
            return_value=(2, 2, self.rgba, self.depth, self.seg),
        )

    @patch("environment.camera.p.getLinkState", return_value=((0, 0, 1), (0, 0, 0, 1)))
    def test_disabled_settings_preserve_render_call_and_rgb_view(self, _link_state):
        for camera in self._cameras():
            for return_seg in (True, False):
                with self.subTest(camera=type(camera).__name__, return_seg=return_seg):
                    self.assertEqual(camera.render_options, {})
                    self.assertEqual(camera.rgb_brightness, 1.0)
                    with self._mock_capture() as render:
                        rgb, depth, seg = self._capture(camera, return_seg=return_seg)
                    expected_kwargs = {"renderer": p.ER_BULLET_HARDWARE_OPENGL}
                    if not return_seg:
                        expected_kwargs["flags"] = p.ER_NO_SEGMENTATION_MASK
                    self.assertEqual(render.call_args.kwargs, expected_kwargs)
                    self.assertEqual(render.call_args.args[:2], (2, 2))
                    self.assertEqual(render.call_args.args[3], camera.projection_matrix)
                    if isinstance(camera, Camera):
                        self.assertEqual(render.call_args.args[2], camera.view_matrix)
                    np.testing.assert_array_equal(rgb, self.rgba[:, :, :3])
                    self.assertTrue(np.shares_memory(rgb, self.rgba))
                    self.assertEqual(rgb.dtype, np.uint8)
                    self.assertIs(depth, self.depth)
                    self.assertIs(seg, self.seg if return_seg else None)

    @patch("environment.camera.p.getLinkState", return_value=((0, 0, 1), (0, 0, 0, 1)))
    def test_lighting_reaches_both_cameras_with_and_without_segmentation(self, _link_state):
        options = {"lightDirection": [0.2, -0.3, 1.0], "shadow": 1}
        for camera in self._cameras():
            camera.render_options = options.copy()
            camera.default_renderer = p.ER_TINY_RENDERER
            for return_seg in (True, False):
                with self.subTest(camera=type(camera).__name__, return_seg=return_seg):
                    with self._mock_capture() as render:
                        self._capture(camera, renderer=None, return_seg=return_seg)
                    expected_kwargs = {**options, "renderer": p.ER_TINY_RENDERER}
                    if not return_seg:
                        expected_kwargs["flags"] = p.ER_NO_SEGMENTATION_MASK
                    self.assertEqual(render.call_args.kwargs, expected_kwargs)
                    self.assertEqual(camera.render_options, options)

    @patch("environment.camera.p.getLinkState", return_value=((0, 0, 1), (0, 0, 0, 1)))
    def test_episode_brightness_only_changes_rgb_and_preserves_geometry(self, _link_state):
        original_rgba = self.rgba.copy()
        for camera in self._cameras():
            projection = tuple(camera.projection_matrix)
            with self._mock_capture() as render:
                baseline = self._capture(camera)
                baseline_geometry = render.call_args.args
                for brightness in (0.6, 1.4):
                    with self.subTest(camera=type(camera).__name__, brightness=brightness):
                        camera.rgb_brightness = brightness
                        first = self._capture(camera)
                        second = self._capture(camera)
                        expected = np.clip(
                            original_rgba[:, :, :3].astype(np.float32) * brightness, 0, 255
                        ).astype(np.uint8)
                        np.testing.assert_array_equal(first[0], expected)
                        np.testing.assert_array_equal(second[0], expected)
                        self.assertEqual(first[0].dtype, np.uint8)
                        self.assertEqual(first[0].shape, (2, 2, 3))
                        self.assertFalse(np.shares_memory(first[0], self.rgba))
                        self.assertIs(first[1], baseline[1])
                        self.assertIs(first[2], baseline[2])
                        self.assertEqual(render.call_args.args, baseline_geometry)
                        self.assertEqual(tuple(camera.projection_matrix), projection)
                        np.testing.assert_array_equal(self.rgba, original_rgba)

    def test_render_options_are_not_shared_between_cameras(self):
        fixed, linked = self._cameras()
        fixed.render_options["shadow"] = 1
        self.assertEqual(linked.render_options, {})
        another_fixed, another_linked = self._cameras()
        self.assertEqual(another_fixed.render_options, {})
        self.assertEqual(another_linked.render_options, {})

    def test_tiny_renderer_lighting_changes_rgb_without_changing_depth_or_labels(self):
        client = p.connect(p.DIRECT)
        self.addCleanup(p.disconnect, client)
        visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=(0.4, 0.4, 0.4),
            rgbaColor=(0.8, 0.5, 0.2, 1), physicsClientId=client,
        )
        p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=visual, basePosition=(0, 0, 0.4),
            physicsClientId=client,
        )
        camera = Camera((2, -2, 2), (0, 0, 0.4), 0.1, 10, (64, 64), 55, up_vector=(0, 0, 1))
        camera.default_renderer = p.ER_TINY_RENDERER
        camera.render_options = {"lightDirection": [1, 0, 1], "shadow": 1, "physicsClientId": client}
        first = camera.get_cam_img(renderer=None)
        camera.render_options["lightDirection"] = [-1, 0, 1]
        second = camera.get_cam_img(renderer=None)
        self.assertTrue(np.any(first[0] != second[0]))
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[2], second[2])


if __name__ == "__main__":
    unittest.main()
