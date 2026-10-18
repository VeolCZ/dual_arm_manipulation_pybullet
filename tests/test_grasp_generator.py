from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from skimage.filters import gaussian
import torch

from grasp_generator import GraspGenerator
from trained_models.GGCNN.models.common import post_process_output as ggcnn_post_process


class TupleNetwork(torch.nn.Module):
    """Produce a broad, unique grasp peak that survives the real detector."""

    def __init__(self, size, center=None):
        super().__init__()
        self.size = size
        self.center = center or (size // 2 - 6, size // 2 + 11)
        self.input = None
        self.grad_enabled = None
        self.output = None

    def forward(self, x):
        self.input = x.detach().clone()
        self.grad_enabled = torch.is_grad_enabled()
        rows, cols = torch.meshgrid(
            torch.arange(self.size, device=x.device),
            torch.arange(self.size, device=x.device),
            indexing="ij",
        )
        quality = torch.exp(-((rows - self.center[0]) ** 2 + (cols - self.center[1]) ** 2) / (2 * 7.0 ** 2))
        angle = torch.full_like(quality, 0.3)
        self.output = tuple(image[None, None] for image in (
            quality, torch.cos(2 * angle), torch.sin(2 * angle), 0.15 + 0.05 * quality,
        ))
        return self.output


class DictNetwork(TupleNetwork):
    def predict(self, x):
        return dict(zip(("pos", "cos", "sin", "width"), super().forward(x)))

    def forward(self, x):
        raise AssertionError("GR_ConvNet must use its dictionary-returning predict method")


class GraspGeneratorTests(unittest.TestCase):
    def generator(self, network="GGCNN", size=300):
        generator = GraspGenerator.__new__(GraspGenerator)
        generator.network = network
        generator.IMG_WIDTH = size
        generator.PIX_CONVERSION = 277 * size / 224
        generator.near, generator.far = 0.1, 10.0
        generator.depth_r = 1
        generator.device = torch.device("cpu")
        generator.net = (DictNetwork if network == "GR_ConvNet" else TupleNetwork)(size)
        generator.fig = Figure(figsize=(6, 4))
        FigureCanvasAgg(generator.fig)
        center = size / 2 - 0.5
        generator.img_to_cam = generator.get_transform_matrix(
            -center / generator.PIX_CONVERSION, center / generator.PIX_CONVERSION,
            0, generator.IMG_ROTATION,
        )
        generator.cam_to_robot_base = generator.get_transform_matrix(0, 0, 1.9, 0)
        return generator

    @staticmethod
    def depth_buffer(metric, near=0.1, far=10.0):
        return ((far - far * near / metric) / (far - near)).astype(np.float32)

    @staticmethod
    def frame(size):
        rgb = np.empty((size, size, 3), dtype=np.uint8)
        rgb[:, :, :] = (40, 90, 160)
        rgb[size // 3:2 * size // 3, size // 3:2 * size // 3] = (190, 120, 25)
        metric = np.full((size, size), 1.115, dtype=np.float32)
        metric[size // 3:2 * size // 3, size // 3:2 * size // 3] = 0.95
        return rgb, metric

    def test_ggcnn_normalizes_metric_depth_and_decodes_native_width_and_peaks(self):
        generator = self.generator()
        rgb, metric = self.frame(300)
        depth = self.depth_buffer(metric)
        grasps, save_name = generator.predict(rgb, depth)

        self.assertEqual(len(grasps), 1)
        self.assertIsNone(save_name)
        self.assertEqual(tuple(generator.net.input.shape), (1, 1, 300, 300))
        self.assertEqual(generator.net.input.device, generator.device)
        self.assertEqual(generator.net.input.dtype, torch.float32)
        self.assertFalse(generator.net.grad_enabled)
        recovered_metric = generator.far * generator.near / (
            generator.far - (generator.far - generator.near) * depth
        )
        expected_input = np.clip(recovered_metric - recovered_metric.mean(), -1, 1)
        np.testing.assert_allclose(generator.net.input.numpy()[0, 0], expected_input, atol=1e-6)
        q_img, angle_img, width_img = ggcnn_post_process(*generator.net.output)
        grasp = grasps[0]
        self.assertEqual(grasp.center, generator.net.center)
        self.assertAlmostEqual(float(grasp.quality), float(q_img[grasp.center]), places=6)
        self.assertAlmostEqual(float(grasp.angle), float(angle_img[grasp.center]), places=6)
        self.assertAlmostEqual(float(grasp.length), float(width_img[grasp.center]), places=5)
        self.assertGreater(grasp.length, 29)

    def test_ggcnn_accepts_metric_depth_without_buffer_conversion(self):
        generator = self.generator()
        rgb, metric = self.frame(300)
        metric += 0.3  # Every distance exceeds 1, so this cannot be a depth buffer.
        grasps, _ = generator.predict(rgb, metric)
        self.assertEqual(len(grasps), 1)
        np.testing.assert_allclose(
            generator.net.input.numpy()[0, 0], np.clip(metric - metric.mean(), -1, 1), atol=1e-6,
        )

    def test_gr_convnet_preserves_rgbd_preprocessing_dictionary_outputs_and_width_scale(self):
        generator = self.generator("GR_ConvNet", size=224)
        rgb, metric = self.frame(224)
        depth = self.depth_buffer(metric)
        grasps, _ = generator.predict(rgb, depth)

        self.assertEqual(len(grasps), 1)
        self.assertEqual(tuple(generator.net.input.shape), (1, 4, 224, 224))
        self.assertFalse(generator.net.grad_enabled)
        recovered_metric = generator.far * generator.near / (
            generator.far - (generator.far - generator.near) * depth
        )
        scaled = recovered_metric * (255.0 / recovered_metric.max())
        expected_depth = np.clip((scaled - scaled.mean()) / 175.0, -1, 1)
        expected_depth /= 255.0
        expected_depth -= expected_depth.mean()
        expected_rgb = rgb.astype(np.float32) / 255.0
        expected_rgb -= expected_rgb.mean()
        np.testing.assert_allclose(generator.net.input.numpy()[0, 0], expected_depth, atol=1e-7)
        np.testing.assert_allclose(generator.net.input.numpy()[0, 1:], expected_rgb.transpose(2, 0, 1), atol=1e-7)
        grasp = grasps[0]
        q, _, _, width = (image.numpy().squeeze() for image in generator.net.output)
        expected_width = gaussian(width * int(generator.MAX_GRASP * generator.PIX_CONVERSION), 1, preserve_range=True)
        self.assertEqual(grasp.center, generator.net.center)
        self.assertAlmostEqual(float(grasp.quality), float(gaussian(q, 1, preserve_range=True)[grasp.center]), places=6)
        self.assertAlmostEqual(float(grasp.length), float(expected_width[grasp.center]), places=6)

    def test_ggcnn_predict_grasp_uses_original_depth_for_world_coordinates(self):
        generator = self.generator()
        rgb, metric = self.frame(300)
        grasps, _ = generator.predict_grasp(rgb, self.depth_buffer(metric))
        self.assertEqual(len(grasps), 1)
        self.assertTrue(np.isfinite(grasps).all())
        row, col = generator.net.center
        center = generator.IMG_WIDTH / 2 - 0.5
        np.testing.assert_allclose(
            grasps[0][:3],
            ((col - center) / generator.PIX_CONVERSION,
             -(row - center) / generator.PIX_CONVERSION, 1.9 - 0.95),
            atol=2e-6,
        )
        self.assertAlmostEqual(grasps[0][3], np.pi / 2 - 0.3, places=6)
        self.assertAlmostEqual(grasps[0][5], generator.DIST_BACKGROUND - 0.95, places=5)

    def test_ggcnn_can_save_a_complete_output_plot(self):
        generator = self.generator()
        rgb, metric = self.frame(300)
        with tempfile.TemporaryDirectory() as directory:
            original = Path.cwd()
            try:
                os.chdir(directory)
                grasps, name = generator.predict(rgb, self.depth_buffer(metric), show_output=True)
                self.assertEqual(len(grasps), 1)
                self.assertIsNotNone(name)
                path = Path(name + ".png")
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 1000)
            finally:
                os.chdir(original)

    def test_mismatched_prediction_grid_is_rejected_before_world_conversion(self):
        generator = self.generator()
        generator.net = TupleNetwork(228)
        rgb, metric = self.frame(300)
        with self.assertRaisesRegex(ValueError, "shape|size|match|300"):
            generator.predict_grasp(rgb, self.depth_buffer(metric))

    def test_unknown_network_is_rejected_before_loading_a_checkpoint(self):
        with self.assertRaises(ValueError):
            GraspGenerator("does-not-exist", SimpleNamespace(), 1, None, network="unknown")

    def test_bundled_ggcnn_checkpoint_loads_and_detects_a_synthetic_object_on_cpu(self):
        root = Path(__file__).resolve().parents[1]
        checkpoint = root / "trained_models/GGCNN/ggcnn_weights_cornell/ggcnn_epoch_23_cornell"
        camera = SimpleNamespace(near=0.1, far=10.0, x=0, y=0, z=1.9)
        with redirect_stdout(StringIO()):
            generator = GraspGenerator(checkpoint, camera, 1, None, network="GGCNN", device="cpu")
        self.assertEqual(generator.IMG_WIDTH, 300)
        self.assertEqual(next(generator.net.parameters()).device, torch.device("cpu"))
        self.assertFalse(generator.net.training)
        rgb, metric = self.frame(300)
        rows, cols = np.mgrid[:300, :300]
        metric[:] = 1.115
        metric[(rows - 150) ** 2 + (cols - 150) ** 2 < 25 ** 2] = 0.95
        grasps, _ = generator.predict(rgb, self.depth_buffer(metric))
        self.assertEqual(len(grasps), 1)
        grasp = grasps[0]
        self.assertLess(np.linalg.norm(np.asarray(grasp.center) - (150, 150)), 25)
        self.assertGreater(grasp.quality, 0.6)
        self.assertGreater(grasp.length, 0)
        self.assertTrue(np.isfinite((*grasp.center, grasp.angle, grasp.quality, grasp.length)).all())


if __name__ == "__main__":
    unittest.main()
