import unittest
import threading
from queue import Queue
from unittest.mock import Mock, patch

import cv2
import numpy as np

from environment.env import Environment
from environment.camera import Camera
from environment.camera_viewer import OpenCVCameraViewer, _run_opencv_viewer, make_camera_mosaics


class CameraStreamingTests(unittest.TestCase):
    def setUp(self):
        # Test streaming without starting a physics server or opening GUI windows.
        self.env = Environment.__new__(Environment)
        self.env.camera_renderer = 123
        self.env.vis = True
        self.env.debug = False
        self.rgb = np.zeros((40, 40, 3), dtype=np.uint8)
        self.rgb[:, :, 0] = 255
        self.env.get_external_cam_img = Mock(return_value=(self.rgb, None, None))

    @patch("cv2.getWindowProperty", return_value=1)
    @patch("cv2.waitKey", return_value=-1)
    @patch("cv2.imshow")
    def test_throttle_leaves_gap_after_slow_capture(self, imshow, wait_key, _visible):
        clock = [10.0]

        def capture(**kwargs):
            clock[0] += 0.4  # Rendering takes longer than the 0.2-second interval.
            return self.rgb, None, None

        self.env.get_external_cam_img.side_effect = capture
        with patch("environment.env.time.monotonic", side_effect=lambda: clock[0]):
            for delay, expected_captures in ((0.0, 1), (0.0, 1), (0.19, 1), (0.02, 2)):
                clock[0] += delay
                self.assertTrue(self.env.visualize_cameras(which=("external",), max_fps=5))
                self.assertEqual(self.env.get_external_cam_img.call_count, expected_captures)
                self.assertEqual(wait_key.call_count, expected_captures)
        self.env.get_external_cam_img.assert_called_with(return_depth=False, return_seg=False, renderer=123)
        # The external tile must convert PyBullet RGB to OpenCV BGR.
        np.testing.assert_array_equal(imshow.call_args.args[1][35, 35], [0, 0, 255])

    @patch("cv2.waitKey", return_value=ord("q"))
    @patch("cv2.imshow")
    def test_quit_returns_without_blocking_even_with_zero_wait(self, _imshow, wait_key):
        self.assertFalse(self.env.visualize_cameras(which=("external",), wait_ms=0))
        wait_key.assert_called_once_with(1)

    @patch("environment.env.p.stepSimulation")
    @patch("environment.env.time.sleep")
    def test_step_dispatch_and_quit_do_not_reopen_stream(self, _sleep, physics_step):
        self.env.enable_camera_streaming(backend="matplotlib", which=("left",), max_fps=2)
        self.env.visualize_cameras_matplotlib = Mock(return_value=True)
        self.env.visualize_cameras = Mock(return_value=False)
        self.env.step_simulation()
        self.env.visualize_cameras_matplotlib.assert_called_once_with(
            which=("left",), show_depth=False, max_fps=2.0, pause_s=0.0, use_blit=True
        )
        self.env.visualize_cameras.assert_not_called()

        with patch("cv2.getBuildInformation", return_value="GUI: QT5"):
            self.env.enable_camera_streaming(backend="opencv")
        self.env.step_simulation()
        self.env.step_simulation()
        self.env.visualize_cameras.assert_called_once()
        self.assertTrue(self.env.visualize_cameras.call_args.kwargs["async_display"])
        self.assertFalse(self.env.auto_stream_cameras)
        self.assertTrue(self.env._camera_stream_configured)
        self.assertEqual(physics_step.call_count, 3)

    @patch("cv2.getBuildInformation", return_value="  GUI: NONE\n")
    def test_headless_opencv_reports_matplotlib_option(self, _build_info):
        with self.assertRaisesRegex(RuntimeError, "stream_backend=matplotlib"):
            self.env.enable_camera_streaming(backend="opencv")

    @patch("cv2.imshow")
    @patch("environment.camera_viewer.make_camera_mosaics")
    @patch("environment.camera_viewer.OpenCVCameraViewer")
    def test_async_submits_raw_frames_without_parent_gui_work(self, worker_class, mosaics, imshow):
        worker = worker_class.return_value
        worker.submit.return_value = True
        self.assertTrue(self.env.visualize_cameras(which=("external",), async_display=True))
        frames, depth = worker.submit.call_args.args
        self.assertIs(frames["external"], self.rgb)
        self.assertIsNone(depth)
        mosaics.assert_not_called()
        imshow.assert_not_called()
        self.env.enable_camera_streaming(enabled=False)
        worker.close.assert_called_once()

    def test_closed_async_worker_is_detected_before_capture_or_throttle(self):
        self.env._cv_camera_worker = Mock()
        self.env._cv_camera_worker.is_alive.return_value = False
        self.assertFalse(self.env.visualize_cameras(which=("external",), max_fps=5, async_display=True))
        self.env.get_external_cam_img.assert_not_called()

    def test_pending_frame_is_replaced_without_waiting(self):
        worker = OpenCVCameraViewer.__new__(OpenCVCameraViewer)
        worker._closed = False
        worker._stop = threading.Event()
        worker._process = Mock()
        worker._process.is_alive.return_value = True
        worker._frames = Queue(maxsize=1)
        for index in range(100):
            self.assertTrue(worker.submit({"external": index}))
        self.assertEqual(worker._frames.qsize(), 1)
        self.assertEqual(worker._frames.get_nowait(), ({"external": 99}, None))

    def test_render_size_and_mosaic_size(self):
        rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        depth = np.zeros((224, 224), dtype=np.float32)
        rgb_mosaic, depth_mosaic = make_camera_mosaics({"external": rgb}, {"external": depth})
        self.assertEqual(rgb_mosaic.shape, (448, 448, 3))
        self.assertEqual(depth_mosaic.shape, (448, 448, 3))

    def test_mosaic_accepts_high_resolution_front_without_resizing_source(self):
        front = np.full((640, 640, 3), 120, dtype=np.uint8)
        topdown = np.full((224, 224, 3), 60, dtype=np.uint8)
        # Put front first to ensure display size does not depend on dict order.
        rgb, depth = make_camera_mosaics(
            {"front": front, "external": topdown},
            {"front": np.ones((640, 640), dtype=np.float32),
             "external": np.ones((224, 224), dtype=np.float32)},
        )
        self.assertEqual(rgb.shape, (448, 448, 3))
        self.assertEqual(depth.shape, (448, 448, 3))
        np.testing.assert_array_equal(rgb[200, 400], [120, 120, 120])
        self.assertEqual(front.shape, (640, 640, 3))
        self.assertTrue(np.all(front == 120))

    @patch("cv2.waitKey", return_value=-1)
    @patch("cv2.imshow")
    def test_front_camera_can_be_selected_for_preview(self, imshow, _wait):
        self.env.get_front_cam_img = Mock(return_value=(self.rgb, None, None))
        self.assertTrue(self.env.visualize_cameras(which=("front",)))
        self.env.get_front_cam_img.assert_called_once_with(return_depth=False, return_seg=False, renderer=123)
        self.env.get_external_cam_img.assert_not_called()
        np.testing.assert_array_equal(imshow.call_args.args[1][35, 75], [0, 0, 255])

    def test_all_five_cameras_fit_in_the_mosaic(self):
        names = ("external", "front", "head", "left", "right")
        frames = {name: np.full((40, 40, 3), value, dtype=np.uint8) for value, name in enumerate(names, 1)}
        rgb, _ = make_camera_mosaics(frames)
        self.assertEqual(rgb.shape, (80, 120, 3))
        for position, value in (((35, 35), 1), ((35, 75), 2), ((35, 115), 3), ((75, 35), 4), ((75, 75), 5)):
            np.testing.assert_array_equal(rgb[position], [value] * 3)
        np.testing.assert_array_equal(rgb[75, 115], [0, 0, 0])

    def test_front_camera_uses_configured_renderer_and_up_direction(self):
        self.env.front_camera = Mock()
        result = self.env.get_front_cam_img(return_depth=False)
        self.assertIs(result, self.env.front_camera.get_cam_img.return_value)
        self.env.front_camera.get_cam_img.assert_called_once_with(return_depth=False, return_seg=False, renderer=123)
        front = Camera(Environment.FRONT_CAMERA_POS, Environment.FRONT_CAMERA_TARGET, .1, 6, Environment.FRONT_CAMERA_SIZE, 55, up_vector=(0, 0, 1))
        self.assertEqual((front.width, front.height), (640, 640))
        self.assertEqual(front.up_vector, (0, 0, 1))
        overhead = Camera((0, 0, 2), (0, 0, 0), .1, 3, (224, 224), 40)
        self.assertEqual(overhead.up_vector, (0, 1, 0))

    @patch("cv2.destroyAllWindows")
    @patch("cv2.getWindowProperty", return_value=1)
    @patch("cv2.waitKey", side_effect=(-1, ord("q")))
    @patch("cv2.imshow")
    @patch("environment.camera_viewer.mp.parent_process", return_value=None)
    def test_worker_processes_quit_even_without_new_frames(self, _parent, imshow, wait_key, _visible, destroy):
        frames = Queue(maxsize=1)
        frames.put_nowait(({"external": self.rgb}, None))
        stop = threading.Event()
        _run_opencv_viewer(frames, stop, "cameras")
        self.assertTrue(stop.is_set())
        self.assertEqual(wait_key.call_count, 2)
        imshow.assert_called_once()
        destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
