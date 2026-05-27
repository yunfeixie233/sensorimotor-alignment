# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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

import fsspec
import pytest

from robo_orchard_lab.dataset.experimental.mcap.batch_decoder import (
    McapBatch2BatchCameraDataConfig,
    McapBatch2BatchCameraDataEncodedConfig,
    McapBatch2BatchFrameTransformConfig,
    McapBatch2BatchJointStateConfig,
    McapBatch2BatchPoseConfig,
    McapBatchDecoderConfig,
    McapBatchDecoders,
)
from robo_orchard_lab.dataset.experimental.mcap.batch_split import (
    McapMessageBatch,
    SplitBatchByTopicArgs,
    SplitBatchByTopics,
    iter_messages_batch,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_decoder import (
    McapDecoderContext,
)
from robo_orchard_lab.dataset.experimental.mcap.reader import (
    McapReader,
)


@pytest.fixture(scope="module")
def example_reader(ROBO_ORCHARD_TEST_WORKSPACE: str):
    mcap_file = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/mcap/RAIL+c3d50939+2023-11-20-17h-48m-24s_image.mcap",
    )
    with fsspec.open(mcap_file, "rb") as f:
        reader = McapReader.make_reader(f)  # type: ignore
        for _, channel in reader.get_summary().channels.items():  # type: ignore
            print(
                f"Channel: {channel.topic}, message_encoding: {channel.message_encoding}"  # noqa: E501
            )
        # for chunk_index in reader.get_summary().chunk_indexes:
        #     print(chunk_index)
        yield reader


@pytest.fixture(scope="module")
def example_msg_batch(example_reader: McapReader):
    # Create a batch of messages from the example reader
    for batch in iter_messages_batch(
        example_reader,
        batch_split=SplitBatchByTopics(
            [
                SplitBatchByTopicArgs(
                    monitor_topic="/observation/cameras/wrist/left/image",
                    min_messages_per_topic=3,
                    max_messages_per_topic=3,
                )
            ]
        ),
    ):
        yield batch
        break


class TestBatch2Data:
    @pytest.mark.parametrize(
        "config",
        [
            McapBatch2BatchJointStateConfig(
                source_topic="/action/robot_state/joints"
            ),
            McapBatch2BatchFrameTransformConfig(
                source_topic="/observation/cameras/wrist/left/tf"
            ),
            McapBatch2BatchCameraDataConfig(
                image_topic="/observation/cameras/wrist/left/image",
                tf_topic="/observation/cameras/wrist/left/tf",
                calib_topic="/observation/cameras/wrist/left/camera_calib",
            ),
        ],
    )
    def test_batch_2_data(
        self,
        example_msg_batch: McapMessageBatch,
        config: McapBatchDecoderConfig,
    ):
        # Create a decoder factory with the converter
        msg_decoder_ctx = McapDecoderContext()
        to_data = config()
        ret = to_data(example_msg_batch, msg_decoder_ctx=msg_decoder_ctx)
        print(ret)


class TestBatch2DataDict:
    @pytest.mark.parametrize(
        "configs",
        [
            {
                "joints": McapBatch2BatchJointStateConfig(
                    source_topic="/action/robot_state/joints"
                ),
                "tf": McapBatch2BatchFrameTransformConfig(
                    source_topic="/observation/cameras/wrist/left/tf"
                ),
                "camera": McapBatch2BatchCameraDataConfig(
                    image_topic="/observation/cameras/wrist/left/image",
                    tf_topic="/observation/cameras/wrist/left/tf",
                    calib_topic="/observation/cameras/wrist/left/camera_calib",
                ),
                "camera_encoded": McapBatch2BatchCameraDataEncodedConfig(
                    image_topic="/observation/cameras/wrist/left/image",
                    tf_topic="/observation/cameras/wrist/left/tf",
                    calib_topic="/observation/cameras/wrist/left/camera_calib",
                ),
                "ee_pose": McapBatch2BatchPoseConfig(
                    source_topic="/action/robot_state/ee_pose"
                ),
            }
        ],
    )
    def test_batch_2_data_dict(
        self,
        example_msg_batch: McapMessageBatch,
        configs: dict[str, McapBatchDecoderConfig],
    ):
        # Create a decoder factory with the converter
        msg_decoder_ctx = McapDecoderContext()
        to_data = McapBatchDecoders(configs)
        ret = to_data(example_msg_batch, msg_decoder_ctx=msg_decoder_ctx)
        for k in configs.keys():
            assert k in ret, f"Key {k} not found in the result"
            print(f"{k}: {type(ret[k])}")
