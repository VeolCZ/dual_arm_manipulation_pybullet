"""Seeded domain randomization of episode appearance without changing physics."""

import hashlib
import math
from pathlib import Path
import random
import shlex
import tempfile

import numpy as np
from PIL import Image
import pybullet as p


class DomainRandomizer:
    """Vary scene appearance and share the sampled settings with all cameras.

    Texture variants are generated lazily outside the object assets. A variant's
    parameters depend on its source, slot and base seed, so a warm texture cache
    does not change the random sequence used to sample subsequent episodes.
    """

    TEXTURE_VARIANTS = 8
    KEEP_ORIGINAL_PROBABILITY = 0.30
    CAMERA_ATTRIBUTES = (
        "camera", "front_camera", "head_camera", "left_arm_camera", "right_arm_camera",
    )
    TABLE_ATTRIBUTES = (
        "tableID", "target_table_id", "target_table_id_right", "UR5Stand_id",
    )

    def __init__(self, env, seed=None):
        self.env = env
        self.base_seed = int(seed) if seed is not None else random.SystemRandom().getrandbits(63)
        self._rng = random.Random(self.base_seed)
        self._episode_rng = None
        self._episode_index = 0
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="pb_domain_randomization_")
        self._original_pixels = {}
        self._source_fingerprints = {}
        self._texture_cache = {}
        self.metadata = None

    def begin_episode(self):
        """Sample and apply table/floor colors, lighting and RGB brightness."""
        if self._temporary_directory is None:
            raise RuntimeError("The domain randomizer is closed")
        episode_seed = self._rng.getrandbits(63)
        self._episode_rng = rng = random.Random(episode_seed)
        self._episode_index += 1
        table_rgba = [rng.uniform(0.30, 0.95) for _ in range(3)] + [1.0]
        floor_rgba = [rng.uniform(0.20, 0.55) for _ in range(3)] + [1.0]
        yaw = math.radians(rng.uniform(-45.0, 45.0))
        elevation = math.radians(rng.uniform(40.0, 70.0))
        light_direction = [
            math.cos(elevation) * math.sin(yaw),
            -math.cos(elevation) * math.cos(yaw),
            math.sin(elevation),
        ]
        render_options = {"lightDirection": light_direction, "shadow": 1}
        brightness = rng.uniform(0.6, 1.4)

        # The continuous tabletop is assembled from four separate bodies.
        # Give every visual link the same material, including the rear stand
        # and both basket platforms, so no white sections remain.
        for attribute in self.TABLE_ATTRIBUTES:
            body_id = getattr(self.env, attribute, None)
            if body_id is not None:
                links = {shape[1] for shape in (p.getVisualShapeData(body_id) or ())}
                for link_index in links:
                    p.changeVisualShape(body_id, link_index, rgbaColor=table_rgba, textureUniqueId=-1)
        p.changeVisualShape(self.env.planeID, -1, rgbaColor=floor_rgba)
        for attribute in self.CAMERA_ATTRIBUTES:
            camera = getattr(self.env, attribute, None)
            if camera is not None:
                camera.render_options = dict(render_options)
                camera.rgb_brightness = brightness
        if getattr(self.env, "vis", False):
            p.configureDebugVisualizer(
                p.COV_ENABLE_SHADOWS,
                1,
                lightPosition=[3.0 * component for component in light_direction],
            )

        self.metadata = {
            "enabled": True,
            "base_seed": self.base_seed,
            "episode_seed": episode_seed,
            "episode_index": self._episode_index,
            "table_rgba": table_rgba,
            "floor_rgba": floor_rgba,
            "render_options": render_options,
            "rgb_brightness": brightness,
            "objects": [],
        }
        return self.metadata

    @staticmethod
    def _find_texture(urdf_path):
        """Find the diffuse map used by the bundled YCB/mesh object assets."""
        asset_directory = Path(urdf_path).expanduser().resolve().parent
        for filename in (
            "texture_map.png", "texture_map.jpg", "texture_map.jpeg",
            "textured_map.png", "textured_map.jpg", "textured_map.jpeg",
            "textured.png", "textured.jpg", "textured.jpeg",
        ):
            candidate = asset_directory / filename
            if candidate.is_file():
                return candidate
        # Some assets name their diffuse image only in a Wavefront material.
        for material in sorted(asset_directory.glob("*.mtl")):
            for line in material.read_text(errors="replace").splitlines():
                fields = shlex.split(line, comments=True)
                if fields and fields[0].lower() == "map_kd" and len(fields) >= 2:
                    candidate = material.parent / " ".join(fields[1:])
                    if candidate.is_file():
                        return candidate.resolve()
        return None

    def _get_texture(self, source, variant):
        key = (str(source), variant)
        if key in self._texture_cache:
            return self._texture_cache[key]
        if variant is None:
            path = source
            parameters = None
        else:
            if str(source) not in self._source_fingerprints:
                self._source_fingerprints[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
            fingerprint = hashlib.sha256(
                f"{self.base_seed}:{self._source_fingerprints[str(source)]}:{variant}".encode("utf-8")
            ).hexdigest()
            rng = random.Random(int(fingerprint, 16))
            parameters = {
                "hue_shift_degrees": rng.uniform(-12.0, 12.0),
                "saturation_multiplier": rng.uniform(0.75, 1.25),
                "value_multiplier": rng.uniform(0.8, 1.2),
            }
            if str(source) not in self._original_pixels:
                with Image.open(source) as image:
                    self._original_pixels[str(source)] = np.asarray(image.convert("HSV")).copy()
            hsv = self._original_pixels[str(source)].astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + parameters["hue_shift_degrees"] * 255.0 / 360.0) % 256.0
            hsv[:, :, 1] *= parameters["saturation_multiplier"]
            hsv[:, :, 2] *= parameters["value_multiplier"]
            hsv = np.clip(hsv, 0.0, 255.0).astype(np.uint8)
            path = Path(self._temporary_directory.name) / f"{fingerprint}.png"
            Image.fromarray(hsv, mode="HSV").convert("RGB").save(path)
        texture_id = p.loadTexture(str(path))
        if texture_id < 0:
            raise RuntimeError(f"Could not load domain randomization texture: {path}")
        record = (texture_id, parameters)
        self._texture_cache[key] = record
        return record

    def apply_object(self, body_id, urdf_path):
        """Apply an original or HSV-varied diffuse texture to a loaded object.

        Applying the original image explicitly is necessary when this body was
        reused from an earlier episode with a randomized texture.
        """
        if self.metadata is None:
            raise RuntimeError("Call begin_episode before applying object textures")
        entry = {"body_id": int(body_id), "asset": str(urdf_path)}
        source = self._find_texture(urdf_path)
        if source is None:
            entry.update({"choice": "unavailable", "source_texture": None})
            self.metadata["objects"].append(entry)
            return entry

        variant = None
        if self._episode_rng.random() >= self.KEEP_ORIGINAL_PROBABILITY:
            variant = self._episode_rng.randrange(self.TEXTURE_VARIANTS)
        texture_id, parameters = self._get_texture(source, variant)
        # A URDF may contain multiple links. Texture changes must cover all of its
        # visual shapes, while leaving their colors, alpha and geometry intact.
        links = {shape[1] for shape in (p.getVisualShapeData(body_id) or ())}
        for link_index in sorted(links):
            p.changeVisualShape(body_id, link_index, textureUniqueId=texture_id)
        entry.update({
            "choice": "original" if variant is None else "hsv_variant",
            "source_texture": str(source),
            "variant_index": variant,
            "hsv_parameters": parameters,
        })
        self.metadata["objects"].append(entry)
        return entry

    def close(self):
        """Remove generated files; loaded textures live with the Bullet scene."""
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None
        self._original_pixels.clear()
        self._source_fingerprints.clear()
        self._texture_cache.clear()
