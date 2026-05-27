# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import pytest

from robo_orchard_lab.distributed.utils import (
    get_dataloader_worker_info,
    get_dist_info,
    is_dist_initialized,
    rank_zero_only,
)


def test_rank_zero_only():
    @rank_zero_only
    def fn():
        return 10

    ret = fn()
    assert ret == 10


def test_is_dist_initialized():
    flag = is_dist_initialized()
    assert not flag


def test_get_dist_info():
    info = get_dist_info()
    assert info.rank == 0
    assert info.world_size == 1


def test_dataloader_worker_info():
    info = get_dataloader_worker_info()
    assert info.rank == 0
    assert info.world_size == 1


if __name__ == "__main__":
    pytest.main(["-s", __file__])
