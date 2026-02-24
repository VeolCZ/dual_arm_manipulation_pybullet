from __future__ import annotations

import argparse
import ast
import shlex
import time
from typing import Literal

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
    p = argparse.ArgumentParser(description="Quick manual test for Behaviors (PyBullet)")
    p.add_argument("--vis", type=_str2bool, default=True, help="Show PyBullet GUI (True/False)")
    p.add_argument("--debug", type=_str2bool, default=False, help="Enable Environment debug (True/False)")
    p.add_argument("--robot", type=str, default="dual", choices=("left", "right", "dual"), help="Which robot to test")
    p.add_argument("--sleep", type=float, default=0.25, help="Sleep seconds between actions (GUI readability)")
    p.add_argument(
        "--interactive",
        type=_str2bool,
        default=False,
        help="Interactive mode: type behaviors in terminal and execute only that behavior",
    )
    return p.parse_args()


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


def main():
    args = parse_args()

    env = _make_env(vis=args.vis, debug=args.debug)
    beh = Behaviors(env)

    try:
        print("\n=== Available behaviors ===")
        beh.list_behaviors()

        robot: RobotSelector = args.robot
        pause = float(args.sleep)

        if bool(args.interactive):
            _interactive_loop(beh, default_robot=robot, pause_s=pause)
            return

        print("\n=== Go to initial pose ===")
        beh.go_to_initial_pose(robot, steps=2000)
        time.sleep(pause)

        print("\n=== Gripper open/close ===")
        beh.open_gripper(robot)
        time.sleep(pause)
        beh.close_gripper(robot, check_contact=False)
        time.sleep(pause)
        beh.open_gripper(robot)
        time.sleep(pause)

        print("\n=== Read joints ===")
        joints = beh.read_all_joints(robot)
        print(f"{robot} joints ({len(joints)}): {[round(v, 4) for v in joints]}")
        time.sleep(pause)

        print("\n=== Set specific joint (small delta) and restore ===")
        idx = 0
        old = beh.get_specific_joint_value(robot, idx)
        beh.set_specific_joint_value(robot, idx, old + 0.05, steps=500)
        time.sleep(pause)
        beh.set_specific_joint_value(robot, idx, old, steps=500)
        time.sleep(pause)

        print("\n=== IK move to pose ===")
        if robot in ("left", "right"):
            x, y, z = _safe_target_xyz(env, robot)
            ok, state = beh.IK_move_to_pose(robot, position_xyz=(x, y, z), yaw=0.0, max_step=1000)
            print(f"IK {robot}: ok={ok} state={state}")
            if state is not None:
                (cur_xyz, _cur_xyzw) = state
                dx = float(cur_xyz[0]) - float(x)
                dy = float(cur_xyz[1]) - float(y)
                dz = float(cur_xyz[2]) - float(z)
                l1 = abs(dx) + abs(dy) + abs(dz)
                print(
                    f"  target=({x:.4f}, {y:.4f}, {z:.4f}) "
                    f"achieved=({cur_xyz[0]:.4f}, {cur_xyz[1]:.4f}, {cur_xyz[2]:.4f}) "
                    f"|err|_1={l1:.6f}"
                )
            time.sleep(pause)

            ee = beh.read_current_ee_pose(robot)
            print(f"EE {robot}: {ee}")

            if args.vis:
                beh.visualize_target_ee_pose(robot, position_xyz=(x, y, z), duration_s=5.0)
                # Add a couple points to trajectory
                beh.visualize_ee_trajectory(robot, position_xyz=(x, y, z), duration_s=10.0)
                beh.visualize_ee_trajectory(robot, position_xyz=(x, y, z + 0.05), duration_s=10.0)
                beh.visualize_ee_trajectory(robot, position_xyz=(x, y, z), duration_s=10.0)

        else:
            lx, ly, lz = _safe_target_xyz(env, "left")
            rx, ry, rz = _safe_target_xyz(env, "right")
            ok, _ = beh.IK_move_to_pose(
                "dual",
                position_xyz=(0.0, 0.0, 0.0),  # ignored for dual
                left_position_xyz=(lx, ly, lz),
                right_position_xyz=(rx, ry, rz),
                yaw=0.0,
                max_step=1000,
            )
            print(f"IK dual: ok={ok}")
            time.sleep(pause)

            ee = beh.read_current_ee_pose("dual")
            print(f"EE dual: {ee}")

            if args.vis:
                beh.visualize_target_ee_pose(
                    "dual",
                    position_xyz=(0.0, 0.0, 0.0),  # ignored for dual
                    left_position_xyz=(lx, ly, lz),
                    right_position_xyz=(rx, ry, rz),
                    duration_s=5.0,
                )

        print("\n=== Object pose (example: target basket) ===")
        try:
            beh.print_object_pose_from_pybullet(getattr(env, "target_id", -1))
        except Exception as exc:
            print(f"Object pose read failed: {exc}")

        print("\nDone.")

    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
