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

import os

import pytest

from robo_orchard_lab.utils.path import in_cwd


def test_in_cwd_changes_and_restores_directory(tmp_path):
    original_dir = os.getcwd()
    sub_dir = tmp_path / "test_dir"
    sub_dir.mkdir()

    assert str(sub_dir) != original_dir

    with in_cwd(str(sub_dir)) as yielded_path:
        assert os.getcwd() == str(sub_dir)
        assert yielded_path == str(sub_dir)

    assert os.getcwd() == original_dir


def test_in_cwd_restores_directory_after_exception(tmp_path):
    original_dir = os.getcwd()
    sub_dir = tmp_path / "exception_dir"
    sub_dir.mkdir()

    with pytest.raises(ValueError, match="Something went wrong"):
        with in_cwd(str(sub_dir)):
            assert os.getcwd() == str(sub_dir)
            raise ValueError("Something went wrong")

    assert os.getcwd() == original_dir


def test_in_cwd_with_nonexistent_directory():
    original_dir = os.getcwd()
    non_existent_dir = "a_directory_that_surely_does_not_exist"

    assert not os.path.exists(non_existent_dir)

    with pytest.raises(FileNotFoundError):
        with in_cwd(non_existent_dir):
            pytest.fail(
                "Should not have entered the context block for a non-existent directory"  # noqa: E501
            )

    assert os.getcwd() == original_dir
