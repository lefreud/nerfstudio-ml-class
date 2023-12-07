
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from hdr_nerf.dataparser import HdrNerfDataparser

HdrNerf = MethodSpecification(
  config=TrainerConfig(
    method_name="nerfstudio",
    pipeline=...
    ...
  ),
  description="Unofficial implementation of HDR-NeRF"
)
HdrNerfDataparser = 

