import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from behaviors import BehaviorInfo, Behaviors
import test_behaviours as runner_module


class BehaviorRunnerTests(unittest.TestCase):
    """Check that the executable suite reports useful, trustworthy results."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output_dir = Path(temporary.name)
        self.behaviors = Mock(spec=Behaviors)
        self.behaviors.env = Mock()
        self.behaviors.env.ee_position_limit = ((-0.5, 0.5), (-0.8, 0.2), (0.8, 1.5))
        self.behaviors.env.GRIPPER_MOVING_HEIGHT = 1.25
        self.behaviors.env._tool0_quat_from_grasp_yaw.return_value = (0.0, 0.0, 0.0, 1.0)
        self.behaviors.env.camera.width = 5
        self.behaviors.env.camera.height = 4
        self.behaviors.go_to_initial_pose.return_value = True
        self.behaviors.read_all_joints.return_value = [0.0] * 6
        self.behaviors.read_current_ee_pose.return_value = {
            "position": (0.25, -0.5, 1.25),
            "orientation_xyzw": (0.0, 0.0, 0.0, 1.0),
        }
        common = {
            "list_behaviors", "get_object_pose_from_pybullet", "print_object_pose_from_pybullet",
            "set_object_pose_in_pybullet", "text_to_speech",
        }

        def dispatch(name, robot="any", **kwargs):
            method = getattr(self.behaviors, name)
            return method(**kwargs) if name in common else method(robot, **kwargs)

        self.behaviors.run.side_effect = dispatch

    def run_checks(self, registry, *, robots=("left", "right", "dual")):
        self.behaviors.available_behaviors.return_value = registry
        with patch.object(Behaviors, "available_behaviors", return_value=registry):
            with contextlib.redirect_stdout(io.StringIO()):
                return runner_module.run_all_behaviors(
                    self.behaviors, robots=robots, output_dir=self.output_dir, pause_s=0.0
                )

    def test_common_checks_run_once_and_only_requested_robots_run(self):
        registry = [
            BehaviorInfo("list_behaviors", "any", "common"),
            BehaviorInfo("read_all_joints", "left", "left"),
            BehaviorInfo("read_all_joints", "right", "right"),
            BehaviorInfo("read_all_joints", "dual", "dual"),
        ]
        with patch.object(runner_module.BehaviorCheckRunner, "check", return_value="checked"):
            results = self.run_checks(registry, robots=("left", "right"))
        self.assertEqual(
            [(result.name, result.robot, result.status) for result in results],
            [("list_behaviors", "any", "PASS"),
             ("read_all_joints", "left", "PASS"),
             ("read_all_joints", "right", "PASS")],
        )
        self.assertTrue(all(result.duration_s >= 0.0 for result in results))

    def test_exception_is_reported_and_remaining_checks_continue(self):
        registry = [
            BehaviorInfo("read_all_joints", "left", "first"),
            BehaviorInfo("read_all_joints", "right", "next"),
        ]
        with patch.object(
            runner_module.BehaviorCheckRunner, "check", side_effect=[RuntimeError("bad sensor"), "checked"]
        ) as check:
            results = self.run_checks(registry)
        self.assertEqual([result.status for result in results], ["FAIL", "PASS"])
        self.assertIn("bad sensor", results[0].detail)
        self.assertEqual(check.call_count, 2)

    def test_registry_entry_without_a_check_fails_instead_of_disappearing(self):
        results = self.run_checks([BehaviorInfo("new_unchecked_behavior", "left", "new")])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "FAIL")
        self.assertIn("new_unchecked_behavior", results[0].detail)

    def test_known_stubs_are_explicitly_skipped(self):
        registry = [
            BehaviorInfo("text_to_speech", "any", "stub"),
            BehaviorInfo("tossing_object", "left", "stub"),
        ]
        results = self.run_checks(registry)
        self.assertEqual([result.status for result in results], ["SKIP", "SKIP"])
        self.assertTrue(all(result.detail for result in results))

    def test_unsuccessful_ik_is_a_failure_for_both_entry_points(self):
        self.behaviors.IK_move_to_pose.return_value = (False, None)
        self.behaviors.IK_move_to_pose_real_time.return_value = (False, None)
        results = self.run_checks([
            BehaviorInfo("IK_move_to_pose", "left", "move"),
            BehaviorInfo("IK_move_to_pose_real_time", "left", "move"),
        ])
        self.assertEqual([result.status for result in results], ["FAIL", "FAIL"])
        self.assertTrue(all("did not converge" in result.detail for result in results))
        self.behaviors.IK_move_to_pose.assert_called()
        self.behaviors.IK_move_to_pose_real_time.assert_called()

    def test_corrupt_camera_dimensions_are_a_failure(self):
        self.behaviors.get_external_cam_img.return_value = (
            np.zeros((4, 5, 3), dtype=np.uint8),
            np.zeros((3, 5), dtype=np.float32),
            np.zeros((4, 5), dtype=np.int32),
        )
        results = self.run_checks([
            BehaviorInfo("get_external_cam_img", "any", "camera"),
        ])
        self.assertEqual(results[0].status, "FAIL")
        self.assertIn("shape", results[0].detail.lower())
        self.behaviors.get_external_cam_img.assert_called_once()

    def test_ik_success_flag_does_not_hide_a_wrong_measured_pose(self):
        self.behaviors.IK_move_to_pose.return_value = (True, None)
        self.behaviors.read_current_ee_pose.return_value = {
            "position": (0.4, -0.5, 1.25),
            "orientation_xyzw": (0.0, 0.0, 0.0, 1.0),
        }
        results = self.run_checks([BehaviorInfo("IK_move_to_pose", "left", "move")])
        self.assertEqual(results[0].status, "FAIL")
        self.assertIn("position error", results[0].detail)

    def test_reached_ik_pose_passes(self):
        self.behaviors.IK_move_to_pose.return_value = (True, None)
        results = self.run_checks([BehaviorInfo("IK_move_to_pose", "left", "move")])
        self.assertEqual(results[0].status, "PASS", results[0].detail)

    def test_no_op_gripper_cannot_pass_by_already_being_at_the_target(self):
        for name, opening in (("open_gripper", 0.085), ("partially_open_gripper", 0.04)):
            with self.subTest(behavior=name):
                with patch.object(runner_module.BehaviorCheckRunner, "_opening", return_value=opening):
                    results = self.run_checks([BehaviorInfo(name, "left", "gripper")])
                self.assertEqual(results[0].status, "FAIL")
                self.assertIn("did not move", results[0].detail)

    def test_main_exits_nonzero_for_failed_checks_and_closes_environment(self):
        args = SimpleNamespace(vis=False, debug=False, robot="all", sleep=0.0,
                               interactive=False, output_dir=None)
        environment = Mock()
        failed = runner_module.BehaviorCheckResult("IK_move_to_pose", "left", "FAIL", "unreachable", 0.1)
        with patch.object(runner_module, "parse_args", return_value=args):
            with patch.object(runner_module, "_make_env", return_value=environment):
                with patch.object(runner_module, "Behaviors", return_value=self.behaviors):
                    with patch.object(runner_module, "run_all_behaviors", return_value=[failed]):
                        self.assertEqual(runner_module.main(), 1)
        environment.close.assert_called_once()

    def test_valid_rgb_depth_segmentation_frames_pass(self):
        self.behaviors.get_external_cam_img.return_value = (
            np.zeros((4, 5, 3), dtype=np.uint8),
            np.ones((4, 5), dtype=np.float32),
            np.zeros((4, 5), dtype=np.int32),
        )
        results = self.run_checks([
            BehaviorInfo("get_external_cam_img", "any", "camera"),
        ])
        self.assertEqual(results[0].status, "PASS", results[0].detail)
        self.behaviors.get_external_cam_img.assert_called_once()


if __name__ == "__main__":
    unittest.main()
