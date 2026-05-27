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

import fsspec
import pytest

from robo_orchard_lab.dataset.experimental.mcap.batch_split import (
    SplitBatchByTopicArgs,
    SplitBatchByTopics,
    iter_messages_batch,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (
    CompressedImage2NumpyConfig,
    NumpyImageMsg,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_decoder import (
    DecoderFactoryWithConverter,
    McapDecoderContext,
)
from robo_orchard_lab.dataset.experimental.mcap.reader import (
    MakeIterMsgArgs,
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


class TestSplitBatchByTopics:
    @pytest.mark.parametrize(
        "split_topics, min_messages_per_topic, max_messages_per_topic, lookahead_duration",  # noqa: E501
        [
            ("/observation/cameras/wrist/left/image", 3, 3, 0),
            ("/observation/cameras/wrist/left/image", 3, 4, 1e9),
            ("/observation/cameras/wrist/left/image", 3, 30, 1e9),
        ],
    )
    def test_split_batch_by_single_topic(
        self,
        example_reader: McapReader,
        split_topics: str,
        min_messages_per_topic: int,
        max_messages_per_topic: int,
        lookahead_duration: int,
    ):
        batch_split = SplitBatchByTopics(
            SplitBatchByTopicArgs(
                monitor_topic=split_topics,
                min_messages_per_topic=min_messages_per_topic,
                max_messages_per_topic=max_messages_per_topic,
                lookahead_duration=lookahead_duration,
            )
        )

        last_batch_cnt = 0

        for batch in iter_messages_batch(
            example_reader,
            batch_split=batch_split,
            do_not_split_same_log_time=False,
        ):
            assert last_batch_cnt == 0, (
                "There should be no batches after the last batch."
            )

            if batch.is_last_batch:
                last_batch_cnt += 1
            if not batch.is_last_batch:
                assert split_topics in batch.message_dict
                topic_batch = batch.message_dict[split_topics]
                assert len(topic_batch) >= min_messages_per_topic
                assert len(topic_batch) <= max_messages_per_topic
                if (
                    len(topic_batch) > min_messages_per_topic
                    and min_messages_per_topic > 0
                ):
                    last_match_frame = topic_batch[min_messages_per_topic - 1]
                    assert (
                        last_match_frame.log_time + lookahead_duration
                        >= topic_batch[-1].log_time
                    ), (
                        "The last message in the batch should be within the "
                        "lookahead duration of the last matched frame."
                    )

        assert last_batch_cnt == 1, (
            "There should be exactly one last batch with fewer messages."
        )

    def test_multi_topics(self, example_reader: McapReader):
        low_frequency_topic = "/observation/cameras/wrist/left/camera_calib"
        high_frequency_topic = "/observation/cameras/wrist/left/image"
        # min_mess
        args = [
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
        batch_split = SplitBatchByTopics(args)
        for i, batch in enumerate(
            iter_messages_batch(
                example_reader,
                batch_split=batch_split,
                do_not_split_same_log_time=False,
            )
        ):
            if not batch.is_last_batch:
                assert len(batch.message_dict[high_frequency_topic]) == 3
            if i == 0:
                # The first batch should contain messages from both topics
                assert len(batch.message_dict[low_frequency_topic]) == 1
            else:
                assert (
                    low_frequency_topic not in batch.message_dict
                    or len(batch.message_dict[low_frequency_topic]) == 0
                )


class TestMcapReader:
    def test_get_summary(self, example_reader: McapReader):
        summary = example_reader.get_summary()
        assert summary is not None
        print(summary)

    def test_decode_message(self, example_reader: McapReader):
        # Test decoding a message from the reader
        factory = DecoderFactoryWithConverter(
            converters={
                "foxglove.CompressedImage": CompressedImage2NumpyConfig(),
            },
        )
        decoder_ctx = McapDecoderContext(
            decoder_factories=[
                factory,
            ]
        )
        cnt = 0
        iter_config = MakeIterMsgArgs(
            topics=["/observation/cameras/wrist/left/image"]
        )
        print("iter_config: ", iter_config)

        for msg_tuple in example_reader.iter_decoded_messages(
            decoder_ctx=decoder_ctx,
            iter_config=iter_config,
        ):
            _schema = msg_tuple.schema
            msg_decoded = msg_tuple.decoded_message
            print(_schema.name)  # type: ignore
            assert isinstance(msg_decoded, NumpyImageMsg)
            assert msg_decoded.data.shape[2] == 3
            print(msg_decoded.frame_id, msg_decoded.timestamp)
            cnt += 1
            if cnt > 5:
                break
        assert cnt > 5

    def test_batch_iter(self, example_reader: McapReader):
        # Test batch iteration over messages
        # iter_config = MakeIterMsgArgs(
        #     topics=["/observation/cameras/wrist/left/image"]
        # )
        split_topics = "/observation/cameras/wrist/left/image"

        batch_split = SplitBatchByTopics(
            SplitBatchByTopicArgs(
                monitor_topic=split_topics,
                min_messages_per_topic=3,
                max_messages_per_topic=3,
            )
        )

        for batch in iter_messages_batch(
            example_reader, batch_split=batch_split
        ):
            print("Batch:", batch.message_dict.keys())
            assert len(batch.message_dict[split_topics]) == 3
            break


class TestMakeIterMsgConfig:
    def test_update_config_with_start_offset(self, example_reader: McapReader):
        reader_start_time = (
            example_reader.get_summary().statistics.message_start_time  # type: ignore
        )

        # if start_offset is set, the update should change the start_time
        start_offset = 1
        config = MakeIterMsgArgs(
            start_offset=start_offset,
        )
        config.update_time_range(reader_start_time)
        assert config.start_time is not None
        assert config.start_time == reader_start_time + start_offset
        assert config.end_time is None

    def test_update_config_with_offset_and_duration(
        self, example_reader: McapReader
    ):
        reader_start_time = (
            example_reader.get_summary().statistics.message_start_time  # type: ignore
        )

        # if start_offset is set, the update should change the start_time
        start_offset = 1
        duration = 10
        config = MakeIterMsgArgs(duration=duration, start_offset=start_offset)
        config.update_time_range(reader_start_time)
        assert config.start_time == reader_start_time + start_offset
        assert config.end_time == config.start_time + duration  # type: ignore

    def test_update_config_with_duration(self, example_reader: McapReader):
        reader_start_time = (
            example_reader.get_summary().statistics.message_start_time  # type: ignore
        )

        # if start_offset is set, the update should change the start_time
        duration = 10
        config = MakeIterMsgArgs(
            duration=duration,
        )
        config.update_time_range(reader_start_time)
        assert config.start_time is not None
        assert config.start_time == reader_start_time
        assert config.end_time == reader_start_time + duration


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
