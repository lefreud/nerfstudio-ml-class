from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Type

import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataparserOutputs,
)
from nerfstudio.data.dataparsers.blender_dataparser import (
    Blender,
    BlenderDataParserConfig,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json


@dataclass
class HdrNerfDataparserOutputs(DataparserOutputs):
    exposures: List[float] = field(default_factory=lambda: [])
    """List of exposures for each frame."""

# todo: add the following dataparsers: one for hdrnerf format, one for momo's format
@dataclass
class HdrNerfDataParser(Blender):
    def __init__(self, config: HdrNerfDataParserConfig):
        super().__init__(config)

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        meta = load_from_json(self.data / f"transforms_{split}.json")
        exposure_dict = load_from_json(self.data / f"exposure_{split}.json")
        image_filenames = []
        poses = []
        exposures = []
        for frame in meta["frames"]:
            exposure_dict_key = frame["file_path"] + ".png"
            fname = self.data / Path(frame["file_path"].replace("./", "") + ".png")
            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            exposures.append(exposure_dict[exposure_dict_key])
        exposures_tensor = torch.tensor([exposures], dtype=torch.float32).T
        poses = np.array(poses).astype(np.float32)

        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor
        # scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))
        # 8x bigger than the original scene box
        scene_box = SceneBox(aabb=torch.tensor([[-12, -12, -12], [12, 12, 12]], dtype=torch.float32))

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
            metadata={"exposures": exposures_tensor}
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor
        )

        return dataparser_outputs


@dataclass
class HdrNerfDataParserConfig(BlenderDataParserConfig):
    """Inspired from Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: HdrNerfDataParser)
    """target class to instantiate"""
