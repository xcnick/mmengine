# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import pytest
from omegaconf import __version__ as oc_version

from mmengine.config import LazyCall as L
from mmengine.config import LazyConfig, instantiate

OC_VERSION = tuple(int(x) for x in oc_version.split('.')[:2])


@dataclass
class ShapeSpec:
    """A simple structure that contains basic shape specification about a
    tensor.

    It is often used as the auxiliary inputs/outputs of models, to complement
    the lack of shape inference ability among pytorch modules.
    """

    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None


def reload_lazy_config(cfg):
    """Save an object by LazyConfig.save and load it back.

    This is used to test that a config still works the same after
    serialization/deserialization.
    """
    with tempfile.TemporaryDirectory(prefix='mmengine') as d:
        fname = os.path.join(d, 'mm_cfg_test.yaml')
        LazyConfig.save(cfg, fname)
        return LazyConfig.load(fname)


class TestClass:

    def __init__(self, int_arg, list_arg=None, dict_arg=None, extra_arg=None):
        self.int_arg = int_arg
        self.list_arg = list_arg
        self.dict_arg = dict_arg
        self.extra_arg = extra_arg

    def __call__(self, call_arg):
        return call_arg + self.int_arg


@pytest.mark.skipIf(OC_VERSION < (2, 1), 'omegaconf version too old')
class TestConstruction:

    def test_basic_construct(self):
        cfg = L(TestClass)(
            int_arg=3,
            list_arg=[10],
            dict_arg={},
            extra_arg=L(TestClass)(int_arg=4, list_arg='${..list_arg}'),
        )

        for x in [cfg, reload_lazy_config(cfg)]:
            obj = instantiate(x)
            assert isinstance(obj, TestClass)
            assert obj.int_arg == 3
            assert obj.extra_arg.int_arg == 4
            assert obj.extra_arg.list_arg == obj.list_arg

            # Test interpolation
            x.extra_arg.list_arg = [5]
            obj = instantiate(x)
            assert isinstance(obj, TestClass)
            assert obj.extra_arg.list_arg == [5]

    def test_instantiate_other_obj(self):
        # do nothing for other obj
        assert instantiate(5), 5
        x = [3, 4, 5]
        assert instantiate(x) == x
        x = TestClass(1)
        assert instantiate(x) is x
        x = {'xx': 'yy'}
        assert instantiate(x) is x

    def test_instantiate_lazy_target(self):
        # _target_ is result of instantiate
        objconf = L(L(len)(int_arg=3))(call_arg=4)
        objconf._target_._target_ = TestClass
        assert instantiate(objconf) == 7

    def test_instantiate_list(self):
        lst = [1, 2, L(TestClass)(int_arg=1)]
        x = L(TestClass)(
            int_arg=lst
        )  # list as an argument should be recursively instantiated
        x = instantiate(x).int_arg
        assert x[:2] == [1, 2]
        assert isinstance(x[2], TestClass)
        assert x[2].int_arg == 1

    def test_instantiate_dataclass(self):
        cfg = L(ShapeSpec)(channels=1, width=3)
        # Test original cfg as well as serialization
        for x in [cfg, reload_lazy_config(cfg)]:
            obj = instantiate(x)
            assert isinstance(obj, ShapeSpec)
            assert obj.channels == 1
            assert obj.height is None

    def test_instantiate_dataclass_as_subconfig(self):
        cfg = L(TestClass)(int_arg=1, extra_arg=ShapeSpec(channels=1, width=3))
        # Test original cfg as well as serialization
        for x in [cfg, reload_lazy_config(cfg)]:
            obj = instantiate(x)
            assert isinstance(obj.extra_arg, ShapeSpec)
            assert obj.extra_arg.channels == 1
            assert obj.extra_arg.height is None

    def test_bad_lazycall(self):
        with pytest.raises(Exception):
            L(3)

    def test_interpolation(self):
        cfg = L(TestClass)(int_arg=3, extra_arg='${int_arg}')

        cfg.int_arg = 4
        obj = instantiate(cfg)
        assert obj.extra_arg == 4

        # Test that interpolation still works after serialization
        cfg = reload_lazy_config(cfg)
        cfg.int_arg = 5
        obj = instantiate(cfg)
        assert obj.extra_arg == 5
