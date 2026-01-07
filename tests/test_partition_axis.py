# Copyright 2025 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for partition axis mappings."""

from eformer.common_types import KV_HEAD_DIM
from eformer.escale.partition.manager import PartitionAxis


def test_kv_head_dim_maps_to_attention_kv_dim_axis():
    assert PartitionAxis._SEMANTIC_MAP[KV_HEAD_DIM] == "attention_kv_dim_axis"
    assert PartitionAxis._STANDARD_TO_GENERATION_ATTR_MAP["attention_kv_dim_axis"] == "decode_attention_kv_dim_axis"
