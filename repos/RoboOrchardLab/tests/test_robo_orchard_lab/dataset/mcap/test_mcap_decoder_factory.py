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
import pyarrow as pa
import pytest
from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
from mcap.reader import make_reader

from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (
    CompressedImage2NumpyConfig,
    NumpyImageMsg,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_decoder import (
    DecoderFactoryWithConverter,
)


class TestDecoderFactoryWithConverter:
    def test_decoder_factory_with_no_converter(
        self, ROBO_ORCHARD_TEST_WORKSPACE: str
    ):
        mcap_file = os.path.join(
            ROBO_ORCHARD_TEST_WORKSPACE,
            "robo_orchard_workspace/mcap/IPRL+7790ec0a+2023-04-21-21h-18m-22s_video.mcap",
        )
        # Create an instance of the DecoderFactoryWithConverter
        factory = DecoderFactoryWithConverter()
        # with fsspec.open(mcap_file, "rb") as f:

        with pa.memory_map(mcap_file, "r") as f:  # type: ignore
            reader = make_reader(
                f,  # type: ignore
                decoder_factories=[
                    factory,
                ],
            )
            print(reader.get_summary())
            # Use the factory to create a decoder
            for (
                _schema,
                _channel,
                _msg_raw,
                msg_decoded,
            ) in reader.iter_decoded_messages(
                topics=["/observation/cameras/wrist/left/image"]
            ):
                msg_decoded: CompressedVideo = msg_decoded
                assert (
                    msg_decoded.DESCRIPTOR.full_name
                    == CompressedVideo.DESCRIPTOR.full_name
                )
                print(_schema.name)  # type: ignore
                print(msg_decoded.frame_id)
                break

    @pytest.mark.parametrize(
        "mcap_file",
        [
            "robo_orchard_workspace/mcap/IPRL+7790ec0a+2023-04-21-21h-18m-22s_image.mcap",
        ],
    )
    def test_decoder_factory_with_to_numpy(
        self, ROBO_ORCHARD_TEST_WORKSPACE: str, mcap_file: str
    ):
        # Create an instance of the DecoderFactoryWithConverter
        mcap_file = os.path.join(ROBO_ORCHARD_TEST_WORKSPACE, mcap_file)
        factory = DecoderFactoryWithConverter(
            converters={
                "foxglove.CompressedImage": CompressedImage2NumpyConfig(),
            },
        )
        with fsspec.open(mcap_file, "rb") as f:
            reader = make_reader(
                f,  # type: ignore
                decoder_factories=[factory],
            )
            print(reader.get_summary())
            cnt = 0
            # Use the factory to create a decoder
            for (
                _schema,
                _channel,
                _msg_raw,
                msg_decoded,
            ) in reader.iter_decoded_messages(
                topics=["/observation/cameras/wrist/left/image"]
            ):
                print(_schema.name)  # type: ignore
                assert isinstance(msg_decoded, NumpyImageMsg)
                assert msg_decoded.data.shape[2] == 3
                print(msg_decoded.frame_id, msg_decoded.timestamp)
                cnt += 1
                if cnt > 5:
                    break
            assert cnt > 0


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
