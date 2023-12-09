from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Type

from nerfstudio.data.dataparsers.base_dataparser import (
    DataparserOutputs,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
    Nerfstudio,
    NerfstudioDataParserConfig,
)


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
        initial = super()._generate_dataparser_outputs(split)
        # todo: change this to use real exposures
        with_exposures = HdrNerfDataparserOutputs(**initial.as_dict(), exposures=[1.0 for _ in initial.image_filenames])
        return with_exposures

@dataclass
class HdrNerfDataParserConfig(NerfstudioDataParserConfig):
    """Inspired from Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: HdrNerfDataParser)
    """target class to instantiate"""
