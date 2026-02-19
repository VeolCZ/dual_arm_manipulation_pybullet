import argparse
import math
import time

import pybullet as p

from environment.utilities import Camera
from environment.env import Environment


def _set_arm_ik(env: Environment, arm: str, target_pos, target_orn):
    if arm == "left":
        eef_id = env.left_eef_id
        joint_names = env.left_controlJoints
    elif arm == "right":
        eef_id = env.right_eef_id
        joint_names = env.right_controlJoints
    else:
        raise ValueError("arm must be 'left' or 'right'")

    joint_poses = p.calculateInverseKinematics(
        bodyUniqueId=env.robot_id,
        endEffectorLinkIndex=eef_id,
        targetPosition=list(target_pos),
        targetOrientation=list(target_orn),
        maxNumIterations=200,
    )

    for name in joint_names:
        joint = env.joints[name]
        idx = env.joint_id_to_movable_index[joint.id]
        pose = float(joint_poses[idx])
        p.setJointMotorControl2(
            env.robot_id,
            joint.id,
            p.POSITION_CONTROL,
            targetPosition=pose,
            force=float(joint.maxForce) * float(env.ARM_FORCE_SCALE),
            maxVelocity=float(joint.maxVelocity),
            positionGain=float(env.ARM_POSITION_GAIN),
            velocityGain=float(env.ARM_VELOCITY_GAIN),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--max-fps", type=float, default=15.0)
    ap.add_argument("--img", type=int, default=224)
    args = ap.parse_args()

    # External camera (kept for backwards compatibility + head camera target sharing).
    cam_z = 1.9
    camera = Camera(
        cam_pos=(0.0, -0.54, cam_z),
        cam_target=(0.0, -0.54, 0.785),
        near=0.2,
        far=2.0,
        size=(args.img, args.img),
        fov=40,
    )

    env = Environment(camera, vis=True, debug=False, finger_length=0.06)
    env.reset_robot()

    # If you want streaming driven by env.step_simulation(), enable it here.
    env.enable_camera_streaming(enabled=True, show_depth=False, max_fps=float(args.max_fps))

    # Put a simple object on the table as a visual reference.
    try:
        env.load_isolated_obj("objects/test_cube1.urdf", mod_orn=False, mod_stiffness=False)
    except Exception:
        pass

    # A simple top-down-ish tool orientation.
    tool_orn = p.getQuaternionFromEuler([0.0, math.pi / 2.0, 0.0])

    start_t = time.time()
    while True:
        t = time.time() - start_t
        if t > float(args.seconds):
            break

        # Small cyclic motions for both wrists.
        left_pos = (0.18 + 0.05 * math.sin(t), -0.52 + 0.05 * math.cos(t), 1.10 + 0.03 * math.sin(0.5 * t))
        right_pos = (-0.18 + 0.05 * math.sin(t + math.pi), -0.52 + 0.05 * math.cos(t + math.pi), 1.10 + 0.03 * math.sin(0.5 * t + 1.0))

        _set_arm_ik(env, "left", left_pos, tool_orn)
        _set_arm_ik(env, "right", right_pos, tool_orn)

        env.step_simulation()

        # Streaming is handled by env.step_simulation() via enable_camera_streaming().

    try:
        p.disconnect()
    except Exception:
        pass


if __name__ == "__main__":
    main()
