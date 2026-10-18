import math
import unittest

import numpy as np
import pybullet as p

from behaviors import Behaviors
from environment.camera import Camera
from environment.env import Environment


class TargetMeshTests(unittest.TestCase):
    def test_default_target_mesh_is_visible_for_both_arms(self):
        client = p.connect(p.DIRECT)
        self.addCleanup(p.disconnect, client)
        behaviors = Behaviors(Environment.__new__(Environment))
        behaviors.visualize_target_ee_pose(
            "dual",
            position_xyz=(0, 0, 0),
            left_position_xyz=(0.15, 0, 0),
            right_position_xyz=(-0.15, 0, 0),
        )

        # TinyRenderer produces no pixels when Bullet silently accepts a missing
        # STL, so check the loaded geometry rather than just its allocated UID.
        view = p.computeViewMatrix((0, -0.8, 0.4), (0, 0, 0.08), (0, 0, 1))
        projection = p.computeProjectionMatrixFOV(55, 1, 0.01, 2)
        image = p.getCameraImage(
            128, 128, view, projection, renderer=p.ER_TINY_RENDERER
        )
        visible_ids = set(np.asarray(image[4]).ravel())
        for arm in ("left", "right"):
            self.assertIn(behaviors._ee_target_mesh_uid[arm], visible_ids)


class ArmMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        camera = Camera(
            (0, -0.54, 1.9), (0, -0.54, 0.785), 0.2, 2, (224, 224), 40
        )
        cls.env = Environment(camera, vis=False, debug=False)
        cls.addClassCleanup(cls.env.close)
        cls.behaviors = Behaviors(cls.env)

    def test_dual_grippers_finish_opening_and_partial_opening(self):
        behaviors, env = self.behaviors, self.env

        def opening(arm):
            joint = env.joints[env.mimicParentName_by_arm[arm]]
            angle = p.getJointState(env.robot_id, joint.id)[0]
            return math.sin(0.715 - angle) * 0.1143 + 0.010

        for arm in ("left", "right"):
            env.set_active_arm(arm)
            env.move_gripper(0, step=360)
        closed = {arm: opening(arm) for arm in ("left", "right")}
        behaviors.open_gripper("dual")
        opened = {arm: opening(arm) for arm in ("left", "right")}
        for arm in ("left", "right"):
            with self.subTest(arm=arm, command="open"):
                self.assertLess(abs(opened[arm] - 0.085), 0.003)
                self.assertGreater(opened[arm] - closed[arm], 0.02)

        behaviors.partially_open_gripper("dual", opening_length=0.04)
        for arm in ("left", "right"):
            with self.subTest(arm=arm, command="partial"):
                partially_opened = opening(arm)
                self.assertLess(abs(partially_opened - 0.04), 0.003)
                self.assertGreater(opened[arm] - partially_opened, 0.02)

    def test_both_ik_targets_reach_tolerance_within_1000_steps(self):
        behaviors = self.behaviors
        behaviors.go_to_initial_pose("dual", steps=2000)
        behaviors.open_gripper("dual")
        behaviors.close_gripper("dual", check_contact=False)
        behaviors.open_gripper("dual")
        initial = behaviors.get_specific_joint_value("left", 0)
        behaviors.set_specific_joint_value("left", 0, initial + 0.05, steps=500)
        behaviors.set_specific_joint_value("left", 0, initial, steps=500)
        targets = {"left": (0.25, -0.5, 1.25), "right": (-0.25, -0.5, 1.25)}
        reached, _ = behaviors.IK_move_to_pose(
            "dual",
            position_xyz=targets["left"],
            left_position_xyz=targets["left"],
            right_position_xyz=targets["right"],
            max_step=1000,
        )
        self.assertTrue(reached)
        for arm, target in targets.items():
            with self.subTest(arm=arm):
                state = behaviors.read_current_ee_pose(arm)
                self.assertLess(np.linalg.norm(np.asarray(state["position"]) - target), 0.005)

    def test_eef_markers_follow_arm_motion_without_constraining_it(self):
        behaviors, env = self.behaviors, self.env
        behaviors.go_to_initial_pose("dual", steps=2000)
        constraint_count = p.getNumConstraints()
        offset = (0.01, -0.02, 0.03)
        rotation = p.getQuaternionFromEuler((0.1, -0.2, 0.3))
        behaviors.enable_eef_mesh_debug(
            "dual",
            parent_to_mesh_xyz=offset,
            parent_to_mesh_quat_xyzw=rotation,
        )
        self.addCleanup(behaviors.enable_eef_mesh_debug, "dual", enabled=False)
        self.assertEqual(p.getNumConstraints(), constraint_count)

        for arm in ("left", "right"):
            with self.subTest(arm=arm):
                initial = behaviors.get_specific_joint_value(arm, 0)
                behaviors.set_specific_joint_value(arm, 0, initial + 0.1, steps=500)
                self.assertGreater(behaviors.get_specific_joint_value(arm, 0) - initial, 0.08)
                link = env.left_eef_id if arm == "left" else env.right_eef_id
                state = p.getLinkState(env.robot_id, link, computeForwardKinematics=True)
                expected = p.multiplyTransforms(state[4], state[5], offset, rotation)
                uid = env._eef_mesh_debug_body_uid[arm]
                actual = p.getBasePositionAndOrientation(uid)
                np.testing.assert_allclose(actual[0], expected[0], atol=1e-6)
                np.testing.assert_allclose(actual[1], expected[1], atol=1e-6)
                self.assertFalse(p.getCollisionShapeData(uid, -1))


if __name__ == "__main__":
    unittest.main()
