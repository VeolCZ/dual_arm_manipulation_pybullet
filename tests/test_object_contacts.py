import os
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import pybullet as p

from environment.env import Environment


class ObjectContactTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        for name in ("PB_OBJECT_LATERAL_FRICTION", "PB_OBJECT_ROLLING_FRICTION", "PB_OBJECT_SPINNING_FRICTION"):
            os.environ.pop(name, None)
        client = p.connect(p.DIRECT)
        self.addCleanup(p.disconnect, client)
        p.setGravity(0, 0, -10)
        p.setTimeStep(1 / 240)
        p.setPhysicsEngineParameter(numSolverIterations=200, enableConeFriction=1, useSplitImpulse=1)
        self.env = Environment.__new__(Environment)
        self.env.vis = self.env.debug = False
        self.env.obj_init_pos = (0, 0)
        self.env.obj_ids, self.env.obj_positions, self.env.obj_orientations = [], [], []
        self.env._set_rendering_enabled = lambda enabled: None
        self.env._refresh_rendering = lambda: None
        self.env._ensure_body_opaque = lambda uid: None

    def test_object_contacts_allow_sliding_under_a_modest_push(self):
        plate_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, 1, .05])
        plate = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plate_shape)
        box_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.02, .02, .02])
        box = p.createMultiBody(baseMass=.1, baseCollisionShapeIndex=box_shape, basePosition=[0, 0, .08])
        for uid in (plate, box):
            self.env._apply_object_dynamics(uid, False)
        for _ in range(120):
            p.stepSimulation()
        start = p.getBasePositionAndOrientation(box)[0][0]
        for _ in range(120):
            position = p.getBasePositionAndOrientation(box)[0]
            p.applyExternalForce(box, -1, [.5, 0, 0], position, p.WORLD_FRAME)
            p.stepSimulation()
        self.assertGreater(p.getBasePositionAndOrientation(box)[0][0] - start, .05)

    def test_high_friction_pads_still_hold_and_release_an_object(self):
        box_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.02, .02, .02])
        box = p.createMultiBody(baseMass=.1, baseCollisionShapeIndex=box_shape, basePosition=[0, 0, .3])
        self.env._apply_object_dynamics(box, False)
        pad_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.005, .04, .04])
        pads = []
        for x in (-.022, .022):
            pad = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=pad_shape, basePosition=[x, 0, .3])
            p.changeDynamics(pad, -1, lateralFriction=2.5, rollingFriction=.1, spinningFriction=.1)
            pads.append(pad)
        for _ in range(240):
            p.stepSimulation()
        self.assertGreater(p.getBasePositionAndOrientation(box)[0][2], .25)
        for pad, x in zip(pads, (-.2, .2)):
            p.resetBasePositionAndOrientation(pad, [x, 0, .3], [0, 0, 0, 1])
        for _ in range(120):
            p.stepSimulation()
        self.assertLess(p.getBasePositionAndOrientation(box)[0][2], .1)

    def test_lifting_from_a_previously_sticky_ycb_pile_releases_neighbours(self):
        table_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[2, 2, .05])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=table_shape, basePosition=[0, 0, .735])
        apply_defaults = self.env._apply_object_dynamics

        def legacy_dynamics(uid, soft):
            apply_defaults(uid, soft)
            p.changeDynamics(uid, -1, lateralFriction=2.5, rollingFriction=.01, spinningFriction=.01, frictionAnchor=1)

        # Preserve a pile layout that produced a hanging pear with the old settings.
        self.env._apply_object_dynamics = legacy_dynamics
        names = ["PottedMeatCan", "Pear", "Strawberry", "MustardBottle", "Banana"]
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(1)
        random.shuffle(names)
        root = Path(__file__).resolve().parents[1]
        load_urdf = p.loadURDF

        def load_from_repo(path, *args, **kwargs):
            return load_urdf(str(root / path), *args, **kwargs)

        info = [(str(root / f"objects/ycb_objects/Ycb{name}/model.urdf"), name == "MustardBottle", name == "Strawberry") for name in names]
        with patch("environment.env.p.loadURDF", side_effect=load_from_repo):
            self.env.create_pile(info)
        self.assertEqual(p.getNumConstraints(), 0)
        self.env._apply_object_dynamics = apply_defaults
        for uid, name in zip(self.env.obj_ids, names):
            apply_defaults(uid, name == "Strawberry")
        held = max(self.env.obj_ids, key=lambda uid: p.getAABB(uid)[1][2])
        position, orientation = p.getBasePositionAndOrientation(held)
        constraint = p.createConstraint(held, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], position,
                                        childFrameOrientation=orientation)
        self.addCleanup(p.removeConstraint, constraint)
        for step in range(240):
            goal = [position[0], position[1], position[2] + .45 * (step + 1) / 240]
            p.changeConstraint(constraint, goal, jointChildFrameOrientation=orientation, maxForce=300)
            p.stepSimulation()
        for _ in range(480):
            p.stepSimulation()
        connected = {held}
        pending = [held]
        while pending:
            for contact in p.getContactPoints(bodyA=pending.pop()):
                other = contact[2]
                if other not in connected:
                    connected.add(other)
                    pending.append(other)
        self.assertEqual(connected.intersection(self.env.obj_ids), {held})


if __name__ == "__main__":
    unittest.main()
