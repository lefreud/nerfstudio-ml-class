from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch

from nerfstudio.data.dataparsers.base_dataparser import (
    DataparserOutputs,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
    Nerfstudio,
    NerfstudioDataParserConfig,
)
from nerfstudio.cameras.cameras import Cameras


@dataclass
class HdrNerfDataparserOutputs(DataparserOutputs):
    exposures: List[float] = field(default_factory=lambda: [])
    """List of exposures for each frame."""

# todo: add the following dataparsers: one for hdrnerf format, one for momo's format
@dataclass
class HdrNerfDataParser(Nerfstudio):
    def __init__(self, config: HdrNerfDataParserConfig):
        super().__init__(config)

    def _generate_dataparser_outputs(self, split="train"):
        outputs = super()._generate_dataparser_outputs(split)
        # todo: change this to use real exposures
        # if outputs.metadata is None:
        #     outputs.metadata = {}
        # outputs.metadata["exposures"] = torch.tensor([[1.0 for _ in outputs.image_filenames]], dtype=torch.float32).T

        if outputs.cameras.metadata is None:
            outputs.cameras.metadata = {}
        outputs.cameras.metadata["exposures"] = torch.tensor([[1.0 for _ in outputs.image_filenames]], dtype=torch.float32).T
        # doesn't seem necessary
        # cameras = Cameras(camera_to_worlds=outputs.cameras.camera_to_worlds,
        #                   fx=outputs.cameras.fx,
        #                   fy=outputs.cameras.fy,
        #                   cx=outputs.cameras.cx,
        #                     cy=outputs.cameras.cy,
        #                   width=outputs.cameras.width,
        #                     height=outputs.cameras.height,
        #                     distortion_params=outputs.cameras.distortion_params,
        #                     camera_type=outputs.cameras.camera_type,
        #                     times=outputs.cameras.times,
        #                   metadata=outputs.cameras.metadata)
        # outputs.cameras = cameras
        # print('outputs bbbb', outputs.cameras.metadata, outputs.metadata)
        return outputs



@dataclass
class HdrNerfDataParserConfig(NerfstudioDataParserConfig):
    """Inspired from Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: HdrNerfDataParser)
    """target class to instantiate"""
