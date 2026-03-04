import os

import pybullet as p
from collections import namedtuple
from attrdict import AttrDict
import functools

# Camera classes live in a dedicated module.
# Kept here as re-exports for backward compatibility.
from environment.camera import Camera, LinkCamera


def setup_sisbot(p, robotID, gripper_type):
    controlJoints = ["shoulder_pan_joint", "shoulder_lift_joint",
                     "elbow_joint", "wrist_1_joint",
                     "wrist_2_joint", "wrist_3_joint",
                     "finger_joint"]
    jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    numJoints = p.getNumJoints(robotID)
    jointInfo = namedtuple("jointInfo",
                           ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                            "controllable"])
    joints = AttrDict()
    for i in range(numJoints):
        info = p.getJointInfo(robotID, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        controllable = True if jointName in controlJoints else False
        info = jointInfo(jointID, jointName, jointType, jointLowerLimit,
                         jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
        if info.type == "REVOLUTE":  # set revolute joint to static
            p.setJointMotorControl2(
                robotID, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        joints[info.name] = info

    # explicitly deal with mimic joints
    def controlGripper(robotID, parent, children, mul, **kwargs):
        controlMode = kwargs.pop("controlMode")
        if controlMode == p.POSITION_CONTROL:
            pose = kwargs.pop("targetPosition")
            # Default gripper motor tuning: stronger + less compliant.
            # (Prevents finger joints from being pushed open by contacts.)
            force_floor = float(os.getenv("PB_GRIPPER_FORCE_N", "800" if gripper_type == "140" else "200"))
            position_gain = float(os.getenv("PB_GRIPPER_PGAIN", "0.6"))
            velocity_gain = float(os.getenv("PB_GRIPPER_VGAIN", "1.0"))
            max_vel = float(os.getenv("PB_GRIPPER_MAX_VEL", "0.3"))

            def _clamp_to_limits(j, q: float) -> float:
                try:
                    ll = float(getattr(j, "lowerLimit", 0.0))
                    ul = float(getattr(j, "upperLimit", -1.0))
                except Exception:
                    return float(q)
                if ul > ll and (ul - ll) < 1e6:
                    return float(min(max(float(q), ll), ul))
                return float(q)

            pose = _clamp_to_limits(parent, float(pose))

            parent_force = max(force_floor, float(parent.maxForce))
            parent_max_vel = max(1e-3, min(max_vel, float(parent.maxVelocity) if float(parent.maxVelocity) > 0 else max_vel))
            # move parent joint
            p.setJointMotorControl2(robotID, parent.id, controlMode, targetPosition=pose,
                                    force=parent_force, maxVelocity=parent_max_vel,
                                    positionGain=position_gain, velocityGain=velocity_gain)
            # move child joints
            for name in children:
                child = children[name]
                childPose = _clamp_to_limits(child, float(pose) * float(mul[child.name]))
                child_force = max(force_floor, float(child.maxForce))
                child_max_vel = max(1e-3, min(max_vel, float(child.maxVelocity) if float(child.maxVelocity) > 0 else max_vel))
                p.setJointMotorControl2(robotID, child.id, controlMode, targetPosition=childPose,
                                        force=child_force, maxVelocity=child_max_vel,
                                        positionGain=position_gain, velocityGain=velocity_gain)
        else:
            raise NotImplementedError(
                "controlGripper does not support \"{}\" control mode".format(controlMode))
        # check if there
        if len(kwargs) != 0:
            raise KeyError("No keys {} in controlGripper".format(
                ", ".join(kwargs.keys())))

    assert gripper_type in ['85', '140']
    mimicParentName = "finger_joint"
    if gripper_type == '85':
        mimicChildren = {"right_outer_knuckle_joint": 1,
                         "left_inner_knuckle_joint": 1,
                         "right_inner_knuckle_joint": 1,
                         "left_inner_finger_joint": -1,
                         "right_inner_finger_joint": -1}
    else:
        mimicChildren = {
            "right_outer_knuckle_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_inner_knuckle_joint": -1,
            "left_inner_finger_joint": 1,
            "right_inner_finger_joint": 1}
    parent = joints[mimicParentName]
    children = AttrDict((j, joints[j])
                        for j in joints if j in mimicChildren.keys())
    controlRobotiqC2 = functools.partial(
        controlGripper, robotID, parent, children, mimicChildren)

    return joints, controlRobotiqC2, controlJoints, mimicParentName


def setup_dual_arm_sisbot(p, robotID, gripper_type: str = "140"):
    """Setup helper for the dual-arm URDF.

    Returns:
        joints: AttrDict joint_name -> jointInfo
        left_arm_joints: list[str] (6 UR joints)
        right_arm_joints: list[str] (6 UR joints)
        control_lgripper: callable(target_position: float)
        control_rgripper: callable(target_position: float)
        joint_id_to_movable_index: dict[int,int] mapping PyBullet joint id -> IK solution index
        link_name_to_id: dict[str,int] mapping child link name -> PyBullet joint id (link index)
    """

    if gripper_type not in ("85", "140"):
        raise ValueError("gripper_type must be '85' or '140'")

    jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    numJoints = p.getNumJoints(robotID)
    jointInfo = namedtuple(
        "jointInfo",
        [
            "id",
            "name",
            "type",
            "lowerLimit",
            "upperLimit",
            "maxForce",
            "maxVelocity",
        ],
    )

    joints = AttrDict()
    joint_id_to_movable_index = {}
    movable_joint_ids = []
    link_name_to_id = {}

    for i in range(numJoints):
        info = p.getJointInfo(robotID, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        ll = float(info[8])
        ul = float(info[9])
        maxF = float(info[10])
        maxV = float(info[11])

        child_link_name = info[12].decode("utf-8")
        link_name_to_id[child_link_name] = jointID

        if info[2] != p.JOINT_FIXED:
            joint_id_to_movable_index[jointID] = len(movable_joint_ids)
            movable_joint_ids.append(jointID)

        ji = jointInfo(jointID, jointName, jointType, ll, ul, maxF, maxV)

        if ji.type == "REVOLUTE":
            p.setJointMotorControl2(robotID, ji.id, p.VELOCITY_CONTROL, targetVelocity=0.0, force=0.0)

        joints[ji.name] = ji

    arm_base = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    left_arm_joints = [f"left_{n}" for n in arm_base]
    right_arm_joints = [f"right_{n}" for n in arm_base]

    def _make_robotiq_mimic_controller(prefix: str):
        gripper_debug = str(os.getenv("PB_GRIPPER_DEBUG", "0")).strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
            "",
        )
        parent_name = prefix + "finger_joint"
        if parent_name not in joints:
            if gripper_debug:
                print(f"[gripper] {prefix}: missing parent joint {parent_name} (no gripper control)")
            return None

        if gripper_type == "85":
            mimic_children = {
                prefix + "right_outer_knuckle_joint": 1,
                prefix + "left_inner_knuckle_joint": 1,
                prefix + "right_inner_knuckle_joint": 1,
                prefix + "left_inner_finger_joint": -1,
                prefix + "right_inner_finger_joint": -1,
            }
        else:
            mimic_children = {
                prefix + "right_outer_knuckle_joint": -1,
                prefix + "left_inner_knuckle_joint": -1,
                prefix + "right_inner_knuckle_joint": -1,
                prefix + "left_inner_finger_joint": 1,
                prefix + "right_inner_finger_joint": 1,
            }

        parent = joints[parent_name]
        children = []
        for cname, mul in mimic_children.items():
            if cname in joints:
                children.append((joints[cname], mul))

        # Prefer a gear-constraint coupling for mimic joints. This keeps the
        # gripper kinematics coherent under contact (prevents links "exploding"
        # or appearing disconnected due to competing motor targets).
        use_gear = str(os.getenv("PB_GRIPPER_USE_GEAR", "1")).strip().lower() not in ("0", "false", "no", "off")
        gear_max_force = float(os.getenv("PB_GRIPPER_GEAR_MAX_FORCE", "100000"))
        gear_erp = float(os.getenv("PB_GRIPPER_GEAR_ERP", "1.0"))
        gear_cids: list[int] = []
        if use_gear and children:
            try:
                for child, mul in children:
                    # Disable direct motor control on mimic joints; let the constraint drive them.
                    try:
                        p.setJointMotorControl2(robotID, child.id, p.VELOCITY_CONTROL, targetVelocity=0.0, force=0.0)
                    except Exception:
                        pass

                    try:
                        jinfo = p.getJointInfo(robotID, int(child.id))
                        axis = jinfo[13]
                    except Exception:
                        axis = (0.0, 0.0, 1.0)

                    cid = p.createConstraint(
                        robotID,
                        int(parent.id),
                        robotID,
                        int(child.id),
                        p.JOINT_GEAR,
                        axis,
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                    )
                    # Negative sign matches the convention used in the original single-arm helper.
                    # ERP helps keep the coupling tight under heavy contact.
                    try:
                        p.changeConstraint(cid, gearRatio=-float(mul), erp=float(gear_erp), maxForce=float(gear_max_force))
                    except Exception:
                        p.changeConstraint(cid, gearRatio=-float(mul), maxForce=float(gear_max_force))
                    gear_cids.append(int(cid))
            except Exception:
                # If gear constraints fail for any reason, fall back to motor-based mimic.
                gear_cids = []

        if gripper_debug:
            child_desc = ", ".join([f"{c.name}(id={c.id},mul={m})" for c, m in children]) if children else "<none>"
            print(
                f"[gripper] {prefix}: parent={parent.name}(id={parent.id}) children={len(children)} "
                f"use_gear={int(use_gear)} gear_cids={len(gear_cids)} gear_max_force={gear_max_force} gear_erp={gear_erp}"
            )
            print(f"[gripper] {prefix}: children: {child_desc}")
            if use_gear and children and not gear_cids:
                print(f"[gripper] {prefix}: WARNING gear requested but no constraints created (falling back to motor mimic)")
            if gear_cids:
                for cid in gear_cids:
                    try:
                        info = p.getConstraintInfo(int(cid))
                        print(f"[gripper] {prefix}: constraint {cid} info={info}")
                    except Exception:
                        print(f"[gripper] {prefix}: constraint {cid} (could not read constraint info)")

        def control(target_position: float):
            target_position = float(target_position)

            # Print once on first control call so we can confirm which path is active.
            nonlocal gripper_debug
            if gripper_debug:
                mode = "gear" if gear_cids else "motor_mimic"
                print(f"[gripper] {prefix}: control() called (mode={mode}, parent_id={parent.id})")
                gripper_debug = False

            # Default gripper motor tuning: stronger + less compliant.
            # Uses a force floor because many URDFs have maxForce=0 for finger joints.
            force_floor = float(os.getenv("PB_GRIPPER_FORCE_N", "1" if gripper_type == "140" else "200"))
            position_gain = float(os.getenv("PB_GRIPPER_PGAIN", "1.0"))
            velocity_gain = float(os.getenv("PB_GRIPPER_VGAIN", "1.0"))
            max_vel = float(os.getenv("PB_GRIPPER_MAX_VEL", "0.75"))

            def _clamp_to_limits(j, q: float) -> float:
                try:
                    ll = float(getattr(j, "lowerLimit", 0.0))
                    ul = float(getattr(j, "upperLimit", -1.0))
                except Exception:
                    return float(q)
                if ul > ll and (ul - ll) < 1e6:
                    return float(min(max(float(q), ll), ul))
                return float(q)

            target_position = _clamp_to_limits(parent, target_position)

            parent_force = max(force_floor, float(parent.maxForce))
            parent_max_vel = max(1e-3, min(max_vel, float(parent.maxVelocity) if float(parent.maxVelocity) > 0 else max_vel))
            p.setJointMotorControl2(
                robotID,
                parent.id,
                p.POSITION_CONTROL,
                targetPosition=target_position,
                force=parent_force,
                maxVelocity=parent_max_vel,
                positionGain=position_gain,
                velocityGain=velocity_gain,
            )
            # If gear constraints are active, they will keep mimic joints in sync.
            # Otherwise, fall back to motor-based mimic control.
            if not gear_cids:
                for child, mul in children:
                    child_target = _clamp_to_limits(child, float(target_position) * float(mul))
                    child_force = max(force_floor, float(child.maxForce))
                    child_max_vel = max(1e-3, min(max_vel, float(child.maxVelocity) if float(child.maxVelocity) > 0 else max_vel))
                    p.setJointMotorControl2(
                        robotID,
                        child.id,
                        p.POSITION_CONTROL,
                        targetPosition=child_target,
                        force=child_force,
                        maxVelocity=child_max_vel,
                        positionGain=position_gain,
                        velocityGain=velocity_gain,
                    )

        return control

    control_lgripper = _make_robotiq_mimic_controller("lgripper_")
    control_rgripper = _make_robotiq_mimic_controller("rgripper_")

    return (
        joints,
        left_arm_joints,
        right_arm_joints,
        control_lgripper,
        control_rgripper,
        joint_id_to_movable_index,
        link_name_to_id,
    )


def setup_sisbot_force(p, robotID, gripper_type):
    controlJoints = ["shoulder_pan_joint", "shoulder_lift_joint",
                     "elbow_joint", "wrist_1_joint",
                     "wrist_2_joint", "wrist_3_joint",
                     "finger_joint"]
    jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    numJoints = p.getNumJoints(robotID)
    jointInfo = namedtuple("jointInfo",
                           ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                            "controllable", "jointAxis", "parentFramePos", "parentFrameOrn"])
    joints = AttrDict()
    for i in range(numJoints):
        info = p.getJointInfo(robotID, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        jointAxis = info[13]
        parentFramePos = info[14]
        parentFrameOrn = info[15]
        controllable = True if jointName in controlJoints else False
        info = jointInfo(jointID, jointName, jointType, jointLowerLimit,
                         jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable,
                         jointAxis, parentFramePos, parentFrameOrn)
        if info.type == "REVOLUTE":  # set revolute joint to static
            p.setJointMotorControl2(
                robotID, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        joints[info.name] = info
    for j in joints:
        print(joints[j])
    # explicitly deal with mimic joints

    def controlGripper(robotID, parent, children, mul, **kwargs):
        controlMode = kwargs.pop("controlMode")
        if controlMode == p.POSITION_CONTROL:
            pose = kwargs.pop("targetPosition")
            # move parent joint
            p.setJointMotorControl2(robotID, parent.id, controlMode, targetPosition=pose,
                                    force=parent.maxForce, maxVelocity=parent.maxVelocity)
            # p.setJointMotorControl2(robotID, parent.id, p.TORQUE_CONTROL,
            #                         force=10, maxVelocity=parent.maxVelocity)
            return
            # move child joints
            for name in children:
                child = children[name]
                childPose = pose * mul[child.name]
                p.setJointMotorControl2(robotID, child.id, controlMode, targetPosition=childPose,
                                        force=child.maxForce, maxVelocity=child.maxVelocity)
        else:
            raise NotImplementedError(
                "controlGripper does not support \"{}\" control mode".format(controlMode))
        # check if there
        if len(kwargs) != 0:
            raise KeyError("No keys {} in controlGripper".format(
                ", ".join(kwargs.keys())))

    assert gripper_type in ['85', '140']
    mimicParentName = "finger_joint"
    if gripper_type == '85':
        mimicChildren = {"right_outer_knuckle_joint": 1,
                         "left_inner_knuckle_joint": 1,
                         "right_inner_knuckle_joint": 1,
                         "left_inner_finger_joint": -1,
                         "right_inner_finger_joint": -1}
    else:
        mimicChildren = {
            "right_outer_knuckle_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_inner_knuckle_joint": -1,
            "left_inner_finger_joint": 1,
            "right_inner_finger_joint": 1}
    parent = joints[mimicParentName]
    children = AttrDict((j, joints[j])
                        for j in joints if j in mimicChildren.keys())
    # Create all the gear constraint
    for name in children:
        child = children[name]
        c = p.createConstraint(robotID, parent.id, robotID, child.id, p.JOINT_GEAR, child.jointAxis,
                               # child.parentFramePos, (0, 0, 0), child.parentFrameOrn, (0, 0, 0))
                               (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
        p.changeConstraint(c, gearRatio=-mimicChildren[name], maxForce=10000)
    controlRobotiqC2 = functools.partial(
        controlGripper, robotID, parent, children, mimicChildren)

    return joints, controlRobotiqC2, controlJoints, mimicParentName



