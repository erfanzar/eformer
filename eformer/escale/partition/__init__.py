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

from .auto_spec import (
    auto_namedsharding,
    auto_partition_spec,
    auto_shard_array,
    convert_sharding_strategy,
    optimize_sharding_for_memory,
    validate_sharding_config,
    vrn_auto_partition_spec,
)
from .constraints import (
    analyze_sharding_strategy,
    create_pattern_based_partition_spec,
    extract_sharding_structure,
    extract_shardings,
    get_axes_size_in_mesh,
    get_corrected_named_sharding,
    get_incontext_mesh,
    get_mesh_axis_names,
    get_mesh_axis_size,
    get_names_from_partition_spec,
    get_partition_spec,
    get_shardings_with_structure,
    get_submesh_device_index,
    make_shard_and_gather_fns,
    match_partition_rules,
    names_in_current_mesh,
    with_sharding_constraint,
)
from .manager import (
    PartitionAxis,
    PartitionManager,
    apply_logical_sharding,
)

__all__ = (
    "PartitionAxis",
    "PartitionManager",
    "analyze_sharding_strategy",
    "apply_logical_sharding",
    "auto_namedsharding",
    "auto_partition_spec",
    "auto_shard_array",
    "convert_sharding_strategy",
    "create_pattern_based_partition_spec",
    "extract_sharding_structure",
    "extract_shardings",
    "get_axes_size_in_mesh",
    "get_corrected_named_sharding",
    "get_incontext_mesh",
    "get_mesh_axis_names",
    "get_mesh_axis_size",
    "get_names_from_partition_spec",
    "get_partition_spec",
    "get_shardings_with_structure",
    "get_submesh_device_index",
    "make_shard_and_gather_fns",
    "match_partition_rules",
    "names_in_current_mesh",
    "optimize_sharding_for_memory",
    "validate_sharding_config",
    "vrn_auto_partition_spec",
    "with_sharding_constraint",
)
