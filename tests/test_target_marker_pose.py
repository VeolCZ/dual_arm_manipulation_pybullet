import unittest

import numpy as np
import pybullet as p

from behaviors import Behaviors
from environment.env import Environment


class TargetMarkerPoseTests(unittest.TestCase):
    def setUp(self):
        client = p.connect(p.DIRECT)
        self.addCleanup(p.disconnect, client)
        self.behaviors = Behaviors(Environment.__new__(Environment))

    def assertMarkerPose(self, arm, position, orientation):
        uid = self.behaviors._ee_target_mesh_uid[arm]
        actual_position, actual_orientation = p.getBasePositionAndOrientation(uid)
        np.testing.assert_allclose(actual_position, position, atol=1e-7)
        # A quaternion and its negation describe the same orientation.
        self.assertAlmostEqual(abs(float(np.dot(actual_orientation, orientation))), 1.0, places=6)

    def test_explicit_world_pose_stays_fixed_during_physics(self):
        orientation = p.getQuaternionFromEuler((0.3, -0.7, 1.2))
        positions = {"left": (0.25, -0.5, 1.25), "right": (-0.25, -0.5, 1.25)}
        for arm, position in positions.items():
            self.behaviors.visualize_target_ee_pose(
                arm, position_xyz=position, orientation_xyzw=orientation
            )
        p.setGravity(0, 0, -10)
        for _ in range(120):
            p.stepSimulation()
        for arm, position in positions.items():
            with self.subTest(arm=arm):
                self.assertMarkerPose(arm, position, orientation)
                self.assertFalse(p.getCollisionShapeData(self.behaviors._ee_target_mesh_uid[arm], -1))
        self.assertEqual(p.getNumConstraints(), 0)

    def test_dual_shared_orientation_applies_to_both_markers(self):
        orientation = p.getQuaternionFromEuler((0.4, -0.2, 0.8))
        self.behaviors.visualize_target_ee_pose(
            "dual",
            position_xyz=(0.25, -0.5, 1.25),
            left_position_xyz=(0.25, -0.5, 1.25),
            right_position_xyz=(-0.25, -0.5, 1.25),
            orientation_xyzw=orientation,
        )
        self.assertMarkerPose("left", (0.25, -0.5, 1.25), orientation)
        self.assertMarkerPose("right", (-0.25, -0.5, 1.25), orientation)

    def test_arm_specific_orientation_overrides_shared_orientation(self):
        shared = p.getQuaternionFromEuler((0.2, 0.3, 0.4))
        override = p.getQuaternionFromEuler((1.1, -0.5, -0.7))
        for arm in ("left", "right"):
            with self.subTest(arm=arm):
                self.behaviors.visualize_target_ee_pose(
                    "dual",
                    position_xyz=(0.25, -0.5, 1.25),
                    left_position_xyz=(0.25, -0.5, 1.25),
                    right_position_xyz=(-0.25, -0.5, 1.25),
                    orientation_xyzw=shared,
                    **{f"{arm}_orientation_xyzw": override},
                )
                self.assertMarkerPose("left", (0.25, -0.5, 1.25), override if arm == "left" else shared)
                self.assertMarkerPose("right", (-0.25, -0.5, 1.25), override if arm == "right" else shared)

    def test_omitted_orientation_keeps_identity(self):
        self.behaviors.visualize_target_ee_pose(
            "dual",
            position_xyz=(0.25, -0.5, 1.25),
            left_position_xyz=(0.25, -0.5, 1.25),
            right_position_xyz=(-0.25, -0.5, 1.25),
        )
        self.assertMarkerPose("left", (0.25, -0.5, 1.25), (0, 0, 0, 1))
        self.assertMarkerPose("right", (-0.25, -0.5, 1.25), (0, 0, 0, 1))


if __name__ == "__main__":
    unittest.main()
