# Copyright (c) OpenMMLab. All rights reserved.
from .config import Config, ConfigDict, DictAction
from .instantiate import instantiate
from .lazy import LazyCall, LazyConfig

__all__ = [
    'Config', 'ConfigDict', 'DictAction', 'LazyCall', 'LazyConfig',
    'instantiate'
]
