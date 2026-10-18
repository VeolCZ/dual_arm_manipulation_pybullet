import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import pybullet as p

from environment.camera import Camera
from environment.episode_recorder import EpisodeRecorder
from environment.domain_randomization import DomainRandomizer


class DomainRandomizationTests(unittest.TestCase):
    def setUp(self):
        client = p.connect(p.DIRECT)
        self.addCleanup(p.disconnect, client)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)

        shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[.4, .4, .02],
                                    rgbaColor=[.8, .8, .8, 1])
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.4, .4, .02])
        table = p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape,
                                 baseCollisionShapeIndex=collision, basePosition=[0, 0, -.04])
        floor = p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape,
                                 baseCollisionShapeIndex=collision, basePosition=[0, 0, -.3])
        self.unrelated_body = p.createMultiBody(baseMass=.4, baseVisualShapeIndex=shape,
                                               baseCollisionShapeIndex=collision,
                                               basePosition=[2, 0, 0])
        self.env = SimpleNamespace(tableID=table, planeID=floor, vis=False)
        for attribute in DomainRandomizer.CAMERA_ATTRIBUTES:
            camera = Camera([0, 0, 1], [0, 0, 0], .1, 2, (32, 32), 50)
            camera.default_renderer = p.ER_TINY_RENDERER
            setattr(self.env, attribute, camera)
        self.env.camera_renderer = p.ER_TINY_RENDERER
        self.env.step_simulation = p.stepSimulation
        self.env.get_external_cam_img = self.env.camera.get_cam_img

        # Four different blocks make texture changes visible without introducing
        # any image resampling or a dependency on the repository's large assets.
        pixels = np.empty((16, 16, 3), dtype=np.uint8)
        pixels[:8, :8] = [190, 35, 40]
        pixels[:8, 8:] = [40, 160, 55]
        pixels[8:, :8] = [35, 65, 180]
        pixels[8:, 8:] = [130, 120, 110]
        self.original_pixels = pixels
        self.texture_path = self.folder / "texture_map.png"
        Image.fromarray(pixels).save(self.texture_path)
        (self.folder / "object.mtl").write_text("newmtl original\nKd 1 1 1\nmap_Kd texture_map.png\n")
        (self.folder / "object.obj").write_text(
            "mtllib object.mtl\n"
            "v -0.08 -0.08 0\nv 0.08 -0.08 0\nv 0.08 0.08 0\nv -0.08 0.08 0\n"
            "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
            "vn 0 0 1\nusemtl original\nf 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n"
        )
        self.urdf_path = self.folder / "object.urdf"
        self.urdf_path.write_text(
            '<robot name="textured_object"><link name="base">'
            '<inertial><mass value=".13"/><inertia ixx=".001" ixy="0" ixz="0" '
            'iyy=".001" iyz="0" izz=".001"/></inertial>'
            '<visual><geometry><mesh filename="object.obj"/></geometry>'
            '<material name="white"><color rgba="1 1 1 1"/></material></visual>'
            '<collision><geometry><box size=".16 .16 .01"/></geometry></collision>'
            '</link></robot>'
        )
        self.object = p.loadURDF(str(self.urdf_path), [0, 0, .03])
        p.changeDynamics(self.object, -1, lateralFriction=.6,
                         rollingFriction=.005, spinningFriction=.005)
        p.resetBaseVelocity(self.object, [.01, .02, .03], [.04, .05, .06])
        self.bodies = (table, floor, self.unrelated_body, self.object)

    def randomizer(self, seed=42):
        randomizer = DomainRandomizer(self.env, seed=seed)
        self.addCleanup(randomizer.close)
        return randomizer

    def physical_state(self):
        return {
            body: (
                p.getDynamicsInfo(body, -1),
                p.getCollisionShapeData(body, -1),
                p.getBasePositionAndOrientation(body),
                p.getBaseVelocity(body),
                p.getAABB(body),
            )
            for body in self.bodies
        }

    def test_visual_changes_preserve_physics_poses_geometry_and_camera_calibration(self):
        before_physics = self.physical_state()
        before_unrelated = p.getVisualShapeData(self.unrelated_body)
        cameras = [getattr(self.env, name) for name in DomainRandomizer.CAMERA_ATTRIBUTES]
        calibration = [
            (camera.view_matrix, camera.projection_matrix, camera.width, camera.height,
             camera.fov, camera.near, camera.far, camera.up_vector)
            for camera in cameras
        ]
        randomizer = self.randomizer()
        metadata = randomizer.begin_episode()
        randomizer.apply_object(self.object, self.urdf_path)

        self.assertEqual(self.physical_state(), before_physics)
        self.assertEqual(p.getVisualShapeData(self.unrelated_body), before_unrelated)
        for index, camera in enumerate(cameras):
            self.assertEqual(
                (camera.view_matrix, camera.projection_matrix, camera.width, camera.height,
                 camera.fov, camera.near, camera.far, camera.up_vector),
                calibration[index],
            )
            self.assertEqual(camera.render_options, metadata["render_options"])
            self.assertEqual(camera.rgb_brightness, metadata["rgb_brightness"])
        np.testing.assert_allclose(p.getVisualShapeData(self.env.tableID)[0][7], metadata["table_rgba"])
        np.testing.assert_allclose(p.getVisualShapeData(self.env.planeID)[0][7], metadata["floor_rgba"])
        self.assertTrue(all(.3 <= value <= .95 for value in metadata["table_rgba"][:3]))
        self.assertTrue(all(.2 <= value <= .55 for value in metadata["floor_rgba"][:3]))
        self.assertGreaterEqual(metadata["rgb_brightness"], .6)
        self.assertLessEqual(metadata["rgb_brightness"], 1.4)

    def test_seed_reproduces_episode_sequence_without_changing_global_rngs(self):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        first, second = self.randomizer(27), self.randomizer(27)

        def sample(randomizer):
            randomizer.begin_episode()
            randomizer.apply_object(self.object, self.urdf_path)
            return copy.deepcopy(randomizer.metadata)

        a1, a2 = sample(first), sample(first)
        b1, b2 = sample(second), sample(second)
        self.assertEqual(a1, b1)
        self.assertEqual(a2, b2)
        self.assertNotEqual(a1["episode_seed"], a2["episode_seed"])
        self.assertNotEqual(a1["table_rgba"], a2["table_rgba"])
        self.assertEqual(random.getstate(), python_state)
        current_numpy = np.random.get_state()
        self.assertEqual(current_numpy[0], numpy_state[0])
        np.testing.assert_array_equal(current_numpy[1], numpy_state[1])
        self.assertEqual(current_numpy[2:], numpy_state[2:])
        self.assertEqual(json.loads(json.dumps(a1)), a1)

    def test_texture_variants_are_real_repeatable_images_and_preserve_original_assets(self):
        source_bytes = self.texture_path.read_bytes()
        variants = []
        metadata = []
        for _ in range(2):
            randomizer = self.randomizer(17)
            randomizer.begin_episode()
            with patch.object(DomainRandomizer, "KEEP_ORIGINAL_PROBABILITY", 0), \
                    patch("environment.domain_randomization.p.loadTexture", wraps=p.loadTexture) as load:
                metadata.append(randomizer.apply_object(self.object, self.urdf_path))
            generated_path = Path(load.call_args.args[0])
            self.assertNotEqual(generated_path, self.texture_path)
            self.assertNotEqual(generated_path.parent, self.folder)
            with Image.open(generated_path) as image:
                variants.append(np.asarray(image).copy())

        self.assertEqual(metadata[0], metadata[1])
        self.assertEqual(metadata[0]["choice"], "hsv_variant")
        np.testing.assert_array_equal(variants[0], variants[1])
        self.assertEqual(variants[0].shape, self.original_pixels.shape)
        self.assertFalse(np.array_equal(variants[0], self.original_pixels))
        self.assertEqual(self.texture_path.read_bytes(), source_bytes)
        # The four texture regions retain their positions and flat colors.
        for row in (0, 8):
            for column in (0, 8):
                block = variants[0][row:row + 8, column:column + 8]
                self.assertEqual(len(np.unique(block.reshape(-1, 3), axis=0)), 1)

    def test_reused_object_can_restore_original_texture_after_a_variant(self):
        def object_rgb():
            # Hold imaging settings fixed while comparing only object pixels;
            # per-episode table colors must not make this check pass by accident.
            self.env.camera.render_options = {}
            self.env.camera.rgb_brightness = 1
            rgb, _, seg = self.env.camera.get_cam_img(renderer=p.ER_TINY_RENDERER)
            pixels = rgb[seg == self.object]
            self.assertGreater(len(pixels), 0)
            return pixels

        before = object_rgb()
        randomizer = self.randomizer()
        randomizer.begin_episode()
        with patch.object(DomainRandomizer, "KEEP_ORIGINAL_PROBABILITY", 0):
            variant = randomizer.apply_object(self.object, self.urdf_path)
        variant_pixels = object_rgb()

        randomizer.begin_episode()
        with patch.object(DomainRandomizer, "KEEP_ORIGINAL_PROBABILITY", 1), \
                patch("environment.domain_randomization.p.loadTexture", wraps=p.loadTexture) as load:
            original = randomizer.apply_object(self.object, self.urdf_path)
        original_pixels = object_rgb()
        self.assertEqual(variant["choice"], "hsv_variant")
        self.assertEqual(original["choice"], "original")
        self.assertFalse(np.array_equal(variant_pixels, before))
        np.testing.assert_array_equal(original_pixels, before)
        self.assertEqual(Path(load.call_args.args[0]), self.texture_path)
        self.assertEqual(randomizer.metadata["objects"], [original])

    def test_missing_texture_keeps_the_visual_and_records_that_it_was_unavailable(self):
        folder = self.folder / "untextured"
        folder.mkdir()
        before = p.getVisualShapeData(self.unrelated_body)
        randomizer = self.randomizer()
        randomizer.begin_episode()
        entry = randomizer.apply_object(self.unrelated_body, folder / "body.urdf")
        self.assertEqual(entry["choice"], "unavailable")
        self.assertIsNone(entry["source_texture"])
        self.assertEqual(before, p.getVisualShapeData(self.unrelated_body))
        self.assertEqual(json.loads(json.dumps(randomizer.metadata)), randomizer.metadata)

    def test_environment_applies_new_choices_to_loaded_and_cached_isolated_objects(self):
        from environment.env import Environment

        env = Environment.__new__(Environment)
        env.__dict__.update(vars(self.env))
        env.obj_ids, env.obj_positions, env.obj_orientations = [], [], []
        env.obj_init_pos = (0, 0)
        env._isolated_obj_cache = {}
        # Isolate object loading/reuse from settling and the GUI. Those paths
        # have separate physical-contact and viewer regression coverage.
        for method in ("_set_rendering_enabled", "_ensure_body_opaque", "_hide_body",
                       "_show_body", "wait_until_still"):
            setattr(env, method, lambda *args: None)
        env._refresh_rendering = env.step_simulation = lambda: None
        randomizer = env._domain_randomizer = DomainRandomizer(env, seed=8)
        self.addCleanup(randomizer.close)
        randomizer.begin_episode()
        with patch.object(DomainRandomizer, "KEEP_ORIGINAL_PROBABILITY", 0):
            env.load_isolated_obj(str(self.urdf_path))
        first_body = env.obj_ids[0]
        self.assertEqual(randomizer.metadata["objects"][0]["body_id"], first_body)
        self.assertEqual(randomizer.metadata["objects"][0]["choice"], "hsv_variant")
        body_count = p.getNumBodies()

        randomizer.begin_episode()
        with patch.object(DomainRandomizer, "KEEP_ORIGINAL_PROBABILITY", 1):
            env.load_isolated_obj(str(self.urdf_path))
        self.assertEqual(env.obj_ids, [first_body])
        self.assertEqual(p.getNumBodies(), body_count)
        self.assertEqual(len(randomizer.metadata["objects"]), 1)
        self.assertEqual(randomizer.metadata["objects"][0]["body_id"], first_body)
        self.assertEqual(randomizer.metadata["objects"][0]["choice"], "original")

    def test_metadata_and_processed_rgb_are_saved_with_episode_recording(self):
        from simulation import GrasppingScenarios

        randomizer = self.env._domain_randomizer = self.randomizer(12)
        randomizer.begin_episode()
        randomizer.apply_object(self.object, self.urdf_path)
        expected_metadata = copy.deepcopy(randomizer.metadata)
        expected_rgb, expected_depth, expected_seg = self.env.get_external_cam_img(
            renderer=p.ER_TINY_RENDERER)
        recorder = self.env._episode_recorder = EpisodeRecorder(
            self.env, self.folder / "episodes", fps=5, which=("external",))
        self.addCleanup(recorder.close)
        scenario = GrasppingScenarios.__new__(GrasppingScenarios)
        scenario.network_model = "GGCNN"
        with scenario._record_episode(self.env, scenario="visual-test"):
            # Later draws must not overwrite the episode's metadata snapshot.
            randomizer.metadata["table_rgba"][0] = 0
        manifest = json.loads((recorder.path / "episode.json").read_text())
        self.assertEqual(manifest["scenario_metadata"]["domain_randomization"], expected_metadata)
        self.assertEqual(manifest["scenario_metadata"]["scenario"], "visual-test")
        with Image.open(recorder.path / "external" / "000000.png") as image:
            np.testing.assert_array_equal(np.asarray(image), expected_rgb)
        # Rendering the same scene twice with fixed settings is stable; depth
        # and segmentation are unaffected by the RGB recording conversion.
        actual_rgb, depth, seg = self.env.get_external_cam_img(renderer=p.ER_TINY_RENDERER)
        np.testing.assert_array_equal(actual_rgb, expected_rgb)
        np.testing.assert_array_equal(depth, expected_depth)
        np.testing.assert_array_equal(seg, expected_seg)


if __name__ == "__main__":
    unittest.main()
