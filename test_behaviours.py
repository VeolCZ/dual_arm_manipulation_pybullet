from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import math
from pathlib import Path
import shlex
from tempfile import TemporaryDirectory
import time
from typing import Literal

import numpy as np
from PIL import Image
import pybullet as pb

from environment.utilities import Camera
from environment.env import Environment

from behaviors import Behaviors


RobotSelector = Literal["left", "right", "dual"]


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("yes", "y", "true", "t", "1", "on"):
        return True
    if s in ("no", "n", "false", "f", "0", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v!r}")


def parse_args():
    p = argparse.ArgumentParser(description="Check all registered Behaviors in PyBullet")
    p.add_argument("--vis", type=_str2bool, default=True, help="Show PyBullet GUI (True/False)")
    p.add_argument("--debug", type=_str2bool, default=False, help="Enable Environment debug (True/False)")
    p.add_argument("--robot", type=str, default="all", choices=("all", "left", "right", "dual"), help="Behavior variants to check (default: all)")
    p.add_argument("--sleep", type=float, default=0.25, help="Sleep seconds between actions (GUI readability)")
    p.add_argument("--output-dir", type=Path, default=None, help="Keep camera check files here (default: temporary files)")
    p.add_argument(
        "--interactive",
        type=_str2bool,
        default=False,
        help="Interactive mode: type behaviors in terminal and execute only that behavior",
    )
    args = p.parse_args()
    if not math.isfinite(args.sleep) or args.sleep < 0:
        p.error("--sleep must be a finite nonnegative number")
    return args


def _parse_value(raw: str):
    """Parse a command value from text.

    Uses `ast.literal_eval` when possible (numbers, tuples, lists, dicts, True/False/None),
    otherwise returns the original string.
    """

    s = str(raw).strip()
    if s == "":
        return ""
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def _bracket_balance_delta(s: str) -> tuple[int, int, int]:
    """Return (paren_delta, bracket_delta, brace_delta) for a string."""
    s = str(s)
    return (
        s.count("(") - s.count(")"),
        s.count("[") - s.count("]"),
        s.count("{") - s.count("}"),
    )


def _merge_key_value_tokens(tokens: list[str]) -> tuple[dict[str, object], list[str]]:
    """Parse key=value tokens, allowing spaces inside bracketed values.

    Example tokens:
        ["joints=(-1,", "2)"] -> {"joints": (-1, 2)}
    """

    kwargs: dict[str, object] = {}
    bad_tokens: list[str] = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if "=" not in tok:
            bad_tokens.append(tok)
            i += 1
            continue

        k, v0 = tok.split("=", 1)
        k = k.strip()
        if not k:
            bad_tokens.append(tok)
            i += 1
            continue

        v_parts = [v0]
        p, b, c = _bracket_balance_delta(v0)

        # If the value starts a bracketed literal and contains spaces, it will be
        # split into multiple tokens by shlex. Merge until balanced.
        while (p > 0 or b > 0 or c > 0) and (i + 1) < len(tokens):
            i += 1
            v_parts.append(tokens[i])
            dp, db, dc = _bracket_balance_delta(tokens[i])
            p += dp
            b += db
            c += dc

        raw_value = " ".join(v_parts).strip()
        kwargs[k] = _parse_value(raw_value)
        i += 1

    return kwargs, bad_tokens


def _interactive_loop(beh: Behaviors, *, default_robot: RobotSelector, pause_s: float):
    print("\nInteractive mode. Type one behavior per line.")
    print("Commands: list, help, quit/exit")
    print("Format: <behavior> [robot] [key=value ...]")
    print(
        "Example: IK_move_to_pose left position_xyz=(0.25,-0.5,1.25) orientation_xyzw=(0,0,0,1) max_step=None"
    )
    print("(You can also use yaw=... instead of orientation_xyzw=...)")

    while True:
        try:
            line = input("behavior> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        line = str(line).strip()
        if not line:
            continue

        if line.lower() in {"quit", "exit", "q"}:
            return

        if line.lower() in {"help", "h", "?", "list"}:
            beh.list_behaviors()
            continue

        try:
            parts = shlex.split(line)
        except Exception as exc:
            print(f"Parse error: {exc}")
            continue

        behavior = parts[0]
        robot = default_robot
        kwargs: dict[str, object] = {}

        rest = parts[1:]
        if rest and rest[0] in {"left", "right", "dual", "any"}:
            # 'any' is accepted for common behaviors.
            robot = rest[0]  # type: ignore[assignment]
            rest = rest[1:]

        kwargs, bad_tokens = _merge_key_value_tokens(rest)
        if bad_tokens:
            print(f"Ignored tokens (expected key=value): {bad_tokens}")

        try:
            out = beh.run(behavior, robot=robot, **kwargs)
            if out is not None:
                print(out)
        except Exception as exc:
            print(f"Error running {behavior!r}: {exc}")

        if pause_s > 0.0:
            time.sleep(float(pause_s))


def _make_env(*, vis: bool, debug: bool) -> Environment:
    # External camera (matches the simulation setup roughly).
    img_size = 224
    cam_z = 1.9
    center_x, center_y, center_z = 0.0, -0.54, cam_z
    camera = Camera(
        (center_x, center_y, center_z),
        (center_x, center_y, 0.785),
        0.2,
        2.0,
        (img_size, img_size),
        40,
    )
    return Environment(camera, vis=bool(vis), debug=bool(debug), hide_gui_panels=True)


def _safe_target_xyz(env: Environment, robot: RobotSelector):
    # Pick a conservative point in the allowed EE box.
    # Prefer left on +x and right on -x to avoid self-collisions.
    xlim, ylim, zlim = env.ee_position_limit
    y = float(min(max(-0.5, ylim[0]), ylim[1]))

    if robot == "left":
        x = float(min(max(0.25, xlim[0]), xlim[1]))
    elif robot == "right":
        x = float(min(max(-0.25, xlim[0]), xlim[1]))
    else:
        # dual handled elsewhere
        x = float(min(max(0.25, xlim[0]), xlim[1]))

    z = float(min(max(getattr(env, "GRIPPER_MOVING_HEIGHT", 1.25), zlim[0]), zlim[1]))
    return x, y, z


class SkipBehavior(Exception):
    """A registered behavior cannot be verified in the current configuration."""


@dataclass(frozen=True)
class BehaviorCheckResult:
    name: str
    robot: str
    status: str
    detail: str
    duration_s: float


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


class BehaviorCheckRunner:
    def __init__(self, behaviors: Behaviors, output_dir: Path):
        self.beh = behaviors
        self.env = behaviors.env
        self.output_dir = Path(output_dir)
        self._object_id = None

    def close(self):
        if self._object_id is not None:
            pb.removeBody(self._object_id)
            self._object_id = None

    def _call(self, name, robot, **kwargs):
        return self.beh.run(name, robot=robot, **kwargs)

    @staticmethod
    def _arms(robot):
        return ('left', 'right') if robot == 'dual' else (robot,)

    def _object(self):
        if self._object_id is None:
            shape = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=(0.015, 0.015, 0.015))
            self._object_id = pb.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=shape,
                basePosition=(0.4, -0.65, 0.9),
            )
        return self._object_id

    def _check_joints(self, robot, expected, tolerance=0.02):
        actual = np.asarray(self.beh.read_all_joints(robot), dtype=float)
        expected = np.asarray(expected, dtype=float)
        _require(actual.shape == expected.shape, 'Joint vector has the wrong length')
        _require(np.isfinite(actual).all(), 'Non-finite joint state')
        error = float(np.max(np.abs(actual - expected)))
        _require(error <= tolerance, f'Maximum joint error {error:.5f} rad exceeds {tolerance} rad')
        return f'max joint error {error:.5f} rad'

    def _validate_camera(self, frames, camera_name):
        rgb, depth, seg = frames
        camera_attr = {'external': 'camera', 'head': 'head_camera',
                       'left_arm': 'left_arm_camera', 'right_arm': 'right_arm_camera'}[camera_name]
        camera = getattr(self.env, camera_attr)
        shape = (camera.height, camera.width)
        rgb, depth, seg = np.asarray(rgb), np.asarray(depth), np.asarray(seg)
        _require(rgb.shape == (*shape, 3), f'RGB shape {rgb.shape}; expected {(*shape, 3)}')
        _require(rgb.dtype == np.uint8, f'RGB dtype is {rgb.dtype}, expected uint8')
        _require(depth.shape == shape, f'Depth shape {depth.shape}; expected {shape}')
        _require(np.isfinite(depth).all(), 'Depth contains non-finite values')
        _require(np.all((depth >= 0) & (depth <= 1)), 'Depth buffer is outside [0, 1]')
        _require(seg.shape == shape, f'Segmentation shape {seg.shape}; expected {shape}')
        _require(np.issubdtype(seg.dtype, np.integer), 'Segmentation is not an integer buffer')
        return f'{shape[1]}x{shape[0]} RGB, depth and segmentation'

    def _check_camera(self, name, robot):
        camera_name = name.removeprefix('get_').removeprefix('save_').removesuffix('_cam_img')
        if name.startswith('get_'):
            return self._validate_camera(
                self._call(name, robot, return_depth=True, return_seg=True), camera_name
            )
        paths = self._call(name, robot, out_dir=str(self.output_dir),
                           return_depth=True, return_seg=True)
        _require(set(paths) == {'rgb_png', 'depth_npy', 'seg_npy'}, 'Missing saved camera outputs')
        for filename in paths.values():
            _require(Path(filename).is_file(), f'Missing camera file: {filename}')
        with Image.open(paths['rgb_png']) as image:
            rgb = np.asarray(image)
        depth = np.load(paths['depth_npy'], allow_pickle=False)
        seg = np.load(paths['seg_npy'], allow_pickle=False)
        return self._validate_camera((rgb, depth, seg), camera_name) + '; saved files reopened'

    def _check_ik(self, name, robot):
        targets = {arm: _safe_target_xyz(self.env, arm) for arm in self._arms(robot)}
        if name == 'IK_move_to_pose_real_time':
            # Exercise another movement instead of checking the previous target again.
            targets = {arm: (x, y, min(z + 0.025, self.env.ee_position_limit[2][1]))
                       for arm, (x, y, z) in targets.items()}
        kwargs = dict(position_xyz=next(iter(targets.values())), yaw=0.0, max_step=1000)
        if robot == 'dual':
            kwargs.update(left_position_xyz=targets['left'], right_position_xyz=targets['right'])
        ok, _ = self._call(name, robot, **kwargs)
        _require(ok, 'IK did not converge within 1000 iterations')
        poses = self.beh.read_current_ee_pose(robot)
        if robot != 'dual':
            poses = {robot: poses}
        errors = []
        for arm, target in targets.items():
            pose = poses[arm]
            position_error = float(np.linalg.norm(np.asarray(pose['position']) - target))
            self.env.set_active_arm(arm)
            target_orn = np.asarray(self.env._tool0_quat_from_grasp_yaw(0.0))
            actual_orn = np.asarray(pose['orientation_xyzw'])
            angular_error = 2 * math.acos(float(np.clip(abs(np.dot(actual_orn, target_orn)), 0, 1)))
            _require(position_error <= 0.005, f'{arm} position error {position_error * 1000:.3f} mm exceeds 5 mm')
            _require(angular_error <= 0.02, f'{arm} orientation error {angular_error:.4f} rad exceeds 0.02 rad')
            errors.append(f'{arm} {position_error * 1000:.3f} mm')
        return ', '.join(errors)

    def _opening(self, arm):
        joint = self.env.joints[self.env.mimicParentName_by_arm[arm]]
        angle = pb.getJointState(self.env.robot_id, joint.id)[0]
        return math.sin(0.715 - angle) * 0.1143 + 0.010

    def _check_gripper(self, name, robot):
        if name == 'close_gripper':
            self.beh.open_gripper(robot)
            before = {arm: self._opening(arm) for arm in self._arms(robot)}
            # False means no object contact; this fixture intentionally has none.
            self._call(name, robot, check_contact=False)
            for arm, opening in before.items():
                _require(self._opening(arm) < opening - 0.015, f'{arm} gripper did not close')
            return 'fingers closed; no object-contact success expected'
        expected = 0.04 if name == 'partially_open_gripper' else 0.085
        if name == 'open_gripper':
            self.beh.partially_open_gripper(robot, opening_length=0.04)
        else:
            self.beh.open_gripper(robot)
        before = {arm: self._opening(arm) for arm in self._arms(robot)}
        self._call(name, robot, opening_length=expected)
        for arm in self._arms(robot):
            actual = self._opening(arm)
            _require(abs(actual - expected) < 0.003,
                     f'{arm} opening {actual:.5f} m did not reach {expected} m')
            _require(abs(actual - before[arm]) > 0.02, f'{arm} fingers did not move')
        return f'opening {expected:.3f} m verified'

    def _check_visualization(self, name, robot):
        arms = self._arms(robot)
        if name == 'enable_eef_mesh_debug':
            original = self.beh.read_all_joints(robot)
            before = {arm: pb.getLinkState(self.env.robot_id, self.beh._eef_link_id(arm))[4]
                      for arm in arms}
            constraints = pb.getNumConstraints()
            try:
                self._call(name, robot, enabled=True)
                target = list(original)
                for index in range(0, len(target), 6):
                    target[index] += 0.03
                self.beh.set_all_joints(robot, target, steps=600)
                for arm in arms:
                    uid = self.env._eef_mesh_debug_body_uid[arm]
                    _require(pb.getVisualShapeData(uid), f'No {arm} EE marker visual shape')
                    marker_pos, marker_orn = pb.getBasePositionAndOrientation(uid)
                    state = pb.getLinkState(self.env.robot_id, self.beh._eef_link_id(arm))
                    _require(np.linalg.norm(np.asarray(state[4]) - before[arm]) > 0.003,
                             f'{arm} arm could not move with the EE marker enabled')
                    _require(np.allclose(marker_pos, state[4], atol=1e-5) and
                             np.allclose(marker_orn, state[5], atol=1e-5),
                             f'{arm} EE marker did not follow the tool')
                _require(pb.getNumConstraints() == constraints, 'EE markers added physics constraints')
            finally:
                self.beh.enable_eef_mesh_debug(robot, enabled=False)
                self.beh.set_all_joints(robot, original, steps=600)
            return 'EE markers follow moving arms without adding constraints'
        if name == 'clear_ee_trajectory':
            self._call(name, robot)
            _require(all(not self.beh._ee_traj_points[arm] for arm in arms), 'Trajectory points were not cleared')
            _require(all(not self.beh._ee_traj_debug_ids[arm] for arm in arms), 'Trajectory drawing IDs were not cleared')
            return 'trajectory points and drawing IDs cleared'
        targets = {arm: _safe_target_xyz(self.env, arm) for arm in arms}
        kwargs = dict(position_xyz=next(iter(targets.values())))
        if robot == 'dual':
            kwargs.update(left_position_xyz=targets['left'], right_position_xyz=targets['right'])
        if name == 'visualize_target_ee_pose':
            orientations = {}
            for arm in arms:
                self.env.set_active_arm(arm)
                orientations[arm] = self.env._tool0_quat_from_grasp_yaw(0.0)
            if robot == 'dual':
                kwargs.update(left_orientation_xyzw=orientations['left'],
                              right_orientation_xyzw=orientations['right'])
            else:
                kwargs['orientation_xyzw'] = orientations[robot]
            self._call(name, robot, **kwargs)
            for arm in arms:
                uid = self.beh._ee_target_mesh_uid[arm]
                actual, orientation = pb.getBasePositionAndOrientation(uid)
                _require(np.allclose(actual, targets[arm]), f'{arm} target marker is at the wrong pose')
                _require(math.isclose(abs(np.dot(orientation, orientations[arm])), 1.0, abs_tol=1e-6),
                         f'{arm} target marker has the wrong orientation')
                _require(pb.getVisualShapeData(uid), f'No {arm} target marker visual shape')
            return 'tool0 target positions, IK orientations and visual shapes verified'
        self.beh.clear_ee_trajectory(robot)
        self._call(name, robot, **kwargs)
        shifted = {arm: (xyz[0], xyz[1], xyz[2] + 0.02) for arm, xyz in targets.items()}
        kwargs['position_xyz'] = next(iter(shifted.values()))
        if robot == 'dual':
            kwargs.update(left_position_xyz=shifted['left'], right_position_xyz=shifted['right'])
        self._call(name, robot, **kwargs)
        _require(all(len(self.beh._ee_traj_points[arm]) == 2 for arm in arms), 'Trajectory points were not added')
        if not self.env.vis:
            raise SkipBehavior('trajectory points verified; debug-line drawing requires the GUI')
        _require(all(any(uid >= 0 for uid in self.beh._ee_traj_debug_ids[arm]) for arm in arms), 'No trajectory debug lines')
        return 'trajectory points and debug lines verified'

    def check(self, name, robot):
        if name in {'tossing_object', 'text_to_speech'}:
            raise SkipBehavior('stub: no throwing implementation' if name == 'tossing_object' else 'stub: no speech backend')
        if name == 'list_behaviors':
            listed = self._call(name, robot)
            _require(listed == self.beh.available_behaviors(), 'Behavior listing does not match the registry')
            return f'{len(listed)} registered entries'
        if name in {'get_object_pose_from_pybullet', 'print_object_pose_from_pybullet', 'set_object_pose_in_pybullet'}:
            body = self._object()
            if name == 'set_object_pose_in_pybullet':
                expected = (0.42, -0.65, 0.92)
                orn = pb.getQuaternionFromEuler((0, 0, 0.3))
                self._call(name, robot, body_id=body, position=expected, orientation_xyzw=orn)
                actual, actual_orn = pb.getBasePositionAndOrientation(body)
                _require(np.allclose(actual, expected) and np.allclose(actual_orn, orn), 'Object pose did not update')
            elif name == 'get_object_pose_from_pybullet':
                pose = self._call(name, robot, body_id=body)
                pos, orn = pb.getBasePositionAndOrientation(body)
                _require(np.allclose(pose['position'], pos) and np.allclose(pose['orientation_xyzw'], orn), 'Object pose readback differs')
            else:
                self._call(name, robot, body_id=body)
            return 'temporary fixture object'
        if name.startswith(('get_', 'save_')) and name.endswith('_cam_img'):
            return self._check_camera(name, robot)
        if name.startswith('go_to_initial'):
            preset = name.removeprefix('go_to_')
            _require(self._call(name, robot, steps=1000), 'Preset motion returned False')
            expected = [v for arm in self._arms(robot) for v in self.beh._PRESET_JOINTS[arm][preset]]
            return self._check_joints(robot, expected)
        if name in {'IK_move_to_pose', 'IK_move_to_pose_real_time'}:
            return self._check_ik(name, robot)
        if name == 'read_all_joints':
            actual = np.asarray(self._call(name, robot), dtype=float)
            expected = [pb.getJointState(self.env.robot_id, self.env.joints[j].id)[0]
                        for arm in self._arms(robot) for j in self.beh._arm_joint_names(arm)]
            _require(actual.shape == (len(expected),) and np.isfinite(actual).all(), 'Invalid joint vector')
            _require(np.allclose(actual, expected), 'Joint readback differs from PyBullet')
            return f'{len(actual)} joint values verified'
        if name == 'get_specific_joint_value':
            joints = self.beh.read_all_joints(robot)
            for index in range(len(joints)):
                value = self._call(name, robot, joint_number=index)
                _require(math.isclose(value, joints[index], abs_tol=1e-6), f'Joint {index} readback differs')
            return f'all {len(joints)} indices verified'
        if name in {'set_all_joints', 'set_specific_joint_value', 'directly_set_all_joints'}:
            original = self.beh.read_all_joints(robot)
            target = list(original)
            target[0] += 0.03
            if robot == 'dual' and name != 'set_specific_joint_value':
                target[6] += 0.03
            try:
                kwargs = dict(joint_number=0, joint_value=target[0], steps=600) if name == 'set_specific_joint_value' else dict(joints=target)
                if name == 'set_all_joints':
                    kwargs['steps'] = 600
                _require(self._call(name, robot, **kwargs), 'Joint command returned False')
                return self._check_joints(robot, target, tolerance=0.003)
            finally:
                self.beh.set_all_joints(robot, original, steps=600)
        if name in {'open_gripper', 'close_gripper', 'partially_open_gripper'}:
            return self._check_gripper(name, robot)
        if name == 'read_current_ee_pose':
            poses = self._call(name, robot)
            if robot != 'dual':
                poses = {robot: poses}
            for arm in self._arms(robot):
                state = pb.getLinkState(self.env.robot_id, self.beh._eef_link_id(arm))
                pose = poses[arm]
                _require(np.allclose(pose['position'], state[0]) and np.allclose(pose['orientation_xyzw'], state[1]), f'{arm} EE readback differs')
            return 'EE positions and orientations verified'
        if name in {'enable_eef_mesh_debug', 'visualize_target_ee_pose', 'visualize_ee_trajectory', 'clear_ee_trajectory'}:
            return self._check_visualization(name, robot)
        raise AssertionError(f'No check implemented for registered behavior {name!r}')


def run_all_behaviors(beh, *, robots, output_dir: Path, pause_s: float = 0.0):
    runner = BehaviorCheckRunner(beh, output_dir)
    results = []
    try:
        for info in beh.available_behaviors():
            if info.robot != 'any' and info.robot not in robots:
                continue
            started = time.perf_counter()
            try:
                detail = runner.check(info.name, info.robot)
                status = 'PASS'
            except SkipBehavior as exc:
                status, detail = 'SKIP', str(exc)
            except Exception as exc:
                status, detail = 'FAIL', f'{type(exc).__name__}: {exc}'
            result = BehaviorCheckResult(info.name, info.robot, status, detail, time.perf_counter() - started)
            results.append(result)
            print(f'[{status}] {info.name:30s} robot={info.robot:5s} ({result.duration_s:.2f}s) {detail}', flush=True)
            if pause_s > 0:
                time.sleep(pause_s)
    finally:
        runner.close()
    counts = {status: sum(result.status == status for result in results) for status in ('PASS', 'FAIL', 'SKIP')}
    print(f"\nChecked {len(results)} registered entries: {counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped.")
    return results


def main():
    args = parse_args()
    env = _make_env(vis=args.vis, debug=args.debug)
    beh = Behaviors(env)
    try:
        if args.interactive:
            _interactive_loop(beh, default_robot='dual' if args.robot == 'all' else args.robot, pause_s=args.sleep)
            return 0
        robots = ('left', 'right', 'dual') if args.robot == 'all' else (args.robot,)
        with TemporaryDirectory(prefix='behavior_checks_') as temporary:
            output_dir = args.output_dir if args.output_dir is not None else Path(temporary)
            results = run_all_behaviors(beh, robots=robots, output_dir=output_dir, pause_s=args.sleep)
        if args.output_dir is not None:
            print(f'Camera check files: {args.output_dir.resolve()}')
        return 1 if any(result.status == 'FAIL' for result in results) else 0
    except KeyboardInterrupt:
        print('\nBehavior checks interrupted.')
        return 130
    finally:
        env.close()


if __name__ == '__main__':
    raise SystemExit(main())
