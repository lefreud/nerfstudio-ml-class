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
        if outputs.cameras.metadata is None:
            outputs.cameras.metadata = {}
        outputs.cameras.metadata["exposures"] = outputs.metadata["exposures"]
        return outputs



@dataclass
class HdrNerfDataParserConfig(NerfstudioDataParserConfig):
    """Inspired from Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: HdrNerfDataParser)
    """target class to instantiate"""
