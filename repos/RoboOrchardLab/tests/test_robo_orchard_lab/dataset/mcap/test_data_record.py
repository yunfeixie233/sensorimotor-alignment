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
from typing import Type

import fsspec
import pytest

from robo_orchard_lab.dataset.experimental.mcap.batch_split import (
    SplitBatchByTopicArgs,
    SplitBatchByTopics,
    iter_messages_batch,
)
from robo_orchard_lab.dataset.experimental.mcap.data_record import (
    McapDataRecordChunk,
    McapDataRecordChunks,
)
from robo_orchard_lab.dataset.experimental.mcap.reader import (
    McapReader,
)


@pytest.fixture(scope="module")
def example_reader(ROBO_ORCHARD_TEST_WORKSPACE: str):
    mcap_file = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/mcap/IPRL+7790ec0a+2023-04-21-21h-18m-22s_image.mcap",
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


class TestDataChunks:
    @pytest.mark.parametrize(
        "rec_chunk_type",
        [McapDataRecordChunk, McapDataRecordChunks],
    )
    def test_read_consistency(
        self,
        example_reader: McapReader,
        rec_chunk_type: Type[McapDataRecordChunk | McapDataRecordChunks],
    ):
        # Create a data record chunk
        low_frequency_topic = "/observation/cameras/wrist/left/camera_calib"
        high_frequency_topic = "/observation/cameras/wrist/left/image"
        batch_split = SplitBatchByTopics(
            [
                SplitBatchByTopicArgs(
                    monitor_topic=low_frequency_topic,
                    min_messages_per_topic=3,
                    max_messages_per_topic=3,
                ),
                SplitBatchByTopicArgs(
                    monitor_topic=high_frequency_topic,
                    min_messages_per_topic=3,
                    max_messages_per_topic=3,
                ),
            ]
        )
        cur_batch = None
        for batch in iter_messages_batch(
            example_reader,
            batch_split=batch_split,
            do_not_split_same_log_time=True,
        ):
            cur_batch = batch
            break

        assert cur_batch is not None

        rec_chunk = rec_chunk_type.from_message_batch(cur_batch)
        print("rec_chunk: ", rec_chunk)
        new_cur_batch = rec_chunk.read(example_reader)
        assert new_cur_batch is not None
        cur_batch.sort()
        new_cur_batch.sort()

        assert len(cur_batch) == len(new_cur_batch)
        for topic in cur_batch.topics:
            assert topic in new_cur_batch
            for msg, new_msg in zip(
                cur_batch[topic], new_cur_batch[topic], strict=True
            ):
                assert msg == new_msg
