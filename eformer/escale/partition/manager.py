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

"""
This module provides classes and functions for managing JAX sharding configurations
and applying sharding constraints within a context.

It includes the `PartitionAxis` class for defining logical-to-physical axis mappings
and the `PartitionManager` context manager for applying these rules.
"""

import dataclasses
import hashlib
import typing as tp

import jax
from jax.sharding import PartitionSpec

from eformer.common_types import (
    BATCH,
    BIAS_HEAD_SEQ,
    BIAS_KV_SEQ,
    DATA_PARALLEL,
    EMBED,
    EMPTY,
    EXPERT,
    EXPERT_GATE,
    EXPERT_PARALLEL,
    FULLY_SHARDED_DATA_PARALLEL,
    GENERATION_MODES,
    HEAD,
    HEAD_DIM,
    KV_HEAD,
    KV_HEAD_DIM,
    KV_LENGTH,
    LENGTH,
    MLP_INTERMEDIATE,
    MODE_DECODE,
    MODE_TRAIN,
    NOT_GIVEN,
    QUERY_LENGTH,
    RUNTIME_MODE_TYPES,
    SEQUENCE_PARALLEL,
    TENSOR_PARALLEL,
    VOCAB,
    AxisType,
    DynamicShardingAxes,
)
from eformer.pytree import PyTree, xTree

from .constraints import get_corrected_named_sharding, with_sharding_constraint


def hash_fn(self) -> int:
    shu = "".join(str(cu) for cu in self.__dict__.values() if isinstance(cu, float | int | float | bool | dict | list))
    return get_safe_hash_int(shu)


def get_safe_hash_int(text, algorithm="md5"):
    try:
        text_str = str(text)
        hash_object = getattr(hashlib, algorithm)(text_str.encode())
        return int.from_bytes(hash_object.digest(), byteorder="big")
    except AttributeError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e
    except Exception as e:
        raise Exception(f"Error generating hash: {e!s}") from e


class PartitionAxis(xTree):
    """
    Configuration for partitioning model axes across a device mesh.

    Defines the mesh dimension names for standard parallelism strategies and maps
    logical model axes to these dimensions. Allows overriding defaults.

    Mesh Dimensions Attributes:
        data_parallel_axis: Name for data parallel mesh dim. Default: "dp".
        fully_sharded_data_parallel_axis: Name for FSDP mesh dim. Default: "fsdp".
        tensor_parallel_axis: Name for tensor parallel mesh dim. Default: "tp".
        sequence_parallel_axis: Name for sequence parallel mesh dim. Default: "sp".
        expert_parallel_axis: Name for expert parallel mesh dim (MoE). Default: "ep".

    Logical Model Axes Attributes:
        Maps logical tensor axes (like batch, sequence, hidden) to one or more
        mesh dimension names defined above, or None if not partitioned.
        Defaults are derived from the standard mesh dimension names but can be
        overridden during instantiation. For example, `head_axis` defaults to
        the value of `tensor_parallel_axis` ('tp').

        batch_axis: Mesh axis for the batch dimension.
        sequence_axis: Mesh axis for the general sequence length dimension.
        query_sequence_axis: Mesh axis for the query sequence length dimension.
        head_axis: Mesh axis for the attention head dimension.
        key_sequence_axis: Mesh axis for the key/value sequence length dimension.
        hidden_state_axis: Mesh axis for the embedding or hidden state dimension.
        mlp_intermediate_axis: Mesh axis for the intermediate dimension in MLP layers.
        vocab_axis: Mesh axis for the vocabulary dimension.
        expert_axis: Mesh axis for the expert dimension.
        expert_gate_axis: Mesh axis for the expert gate dimension.
        attention_dim_axis: Mesh axis for the dimension within each attention head.
        bias_head_sequence_axis: Mesh axis for bias related to head and sequence dimensions.
        bias_key_sequence_axis: Mesh axis for bias related to key/value sequence dimensions.

        decode_batch_axis: Mesh axis for the batch dimension during decoding.
        decode_query_sequence_axis: Mesh axis for the query sequence length during decoding.
        decode_head_axis: Mesh axis for the attention head dimension during decoding.
        decode_key_sequence_axis: Mesh axis for the key/value sequence length during decoding.
        decode_attention_dim_axis: Mesh axis for the dimension within each attention head during decoding.
    """

    data_parallel_axis: str = "dp"
    fully_sharded_data_parallel_axis: str = "fsdp"
    tensor_parallel_axis: str = "tp"
    sequence_parallel_axis: str = "sp"
    expert_parallel_axis: str = "ep"

    batch_axis: AxisType = NOT_GIVEN
    sequence_axis: AxisType = NOT_GIVEN
    query_sequence_axis: AxisType = NOT_GIVEN
    head_axis: AxisType = NOT_GIVEN
    kv_head_axis: AxisType = NOT_GIVEN
    key_sequence_axis: AxisType = NOT_GIVEN
    hidden_state_axis: AxisType = NOT_GIVEN
    mlp_intermediate_axis: AxisType = NOT_GIVEN
    vocab_axis: AxisType = NOT_GIVEN
    expert_axis: AxisType = NOT_GIVEN
    expert_gate_axis: AxisType = None

    attention_dim_axis: AxisType = None
    attention_kv_dim_axis: AxisType = None
    bias_head_sequence_axis: AxisType = None
    bias_key_sequence_axis: AxisType = None

    decode_batch_axis: AxisType = NOT_GIVEN
    decode_query_sequence_axis: AxisType = None
    decode_head_axis: AxisType = NOT_GIVEN
    decode_kv_head_axis: AxisType = NOT_GIVEN
    decode_key_sequence_axis: AxisType = NOT_GIVEN
    decode_attention_dim_axis: AxisType = None
    decode_attention_kv_dim_axis: AxisType = None

    _SEMANTIC_MAP: tp.ClassVar[dict[str, str]] = {
        BATCH: "batch_axis",
        LENGTH: "sequence_axis",
        QUERY_LENGTH: "query_sequence_axis",
        KV_LENGTH: "key_sequence_axis",
        EMBED: "hidden_state_axis",
        HEAD: "head_axis",
        KV_HEAD: "kv_head_axis",
        MLP_INTERMEDIATE: "mlp_intermediate_axis",
        VOCAB: "vocab_axis",
        EXPERT: "expert_axis",
        EXPERT_GATE: "expert_gate_axis",
        HEAD_DIM: "attention_dim_axis",
        KV_HEAD_DIM: "decode_attention_kv_dim_axis",
        BIAS_HEAD_SEQ: "bias_head_sequence_axis",
        BIAS_KV_SEQ: "bias_key_sequence_axis",
        EMPTY: None,
        DATA_PARALLEL: "data_parallel_axis",
        FULLY_SHARDED_DATA_PARALLEL: "fully_sharded_data_parallel_axis",
        TENSOR_PARALLEL: "tensor_parallel_axis",
        SEQUENCE_PARALLEL: "sequence_parallel_axis",
        EXPERT_PARALLEL: "expert_parallel_axis",
    }

    """
	Maps semantic axis name constants (e.g., BATCH) to their corresponding
	attribute names in the PartitionAxis class (e.g., "batch_axis").
	"""

    _STANDARD_TO_GENERATION_ATTR_MAP: tp.ClassVar[dict[str, str]] = {
        "batch_axis": "decode_batch_axis",
        "query_sequence_axis": "decode_query_sequence_axis",
        "key_sequence_axis": "decode_key_sequence_axis",
        "head_axis": "decode_head_axis",
        "kv_head_axis": "decode_kv_head_axis",
        "attention_dim_axis": "decode_attention_dim_axis",
        "attention_kv_dim_axis": "decode_attention_kv_dim_axis",
    }
    """
	Maps standard axis attribute names to their corresponding generation-specific
	attribute names. Used to apply different sharding rules during generation modes.
	"""

    def __post_init__(self):
        """
        Post-initialization hook to resolve default axis values.

        If an axis attribute is set to NOT_GIVEN, its value is resolved based
        on default logic, typically using the standard mesh dimension names.
        """
        resolved_values = {}

        def resolve_field(name, default_logic):
            """Helper to resolve a single field's value if it's NOT_GIVEN."""
            current_value = getattr(self, name)
            if current_value is NOT_GIVEN:
                resolved_values[name] = default_logic()
            elif name not in resolved_values:
                resolved_values[name] = current_value

        def get_resolved(name):
            """Helper to get a field's value, prioritizing resolved values."""
            return resolved_values.get(name, getattr(self, name))

        # Resolve standard axis defaults
        resolve_field(
            "batch_axis",
            lambda: (self.fully_sharded_data_parallel_axis, self.data_parallel_axis),
        )
        resolve_field("sequence_axis", lambda: self.sequence_parallel_axis)
        resolve_field("query_sequence_axis", lambda: self.sequence_parallel_axis)
        # Default qS = S rule
        resolve_field("head_axis", lambda: self.tensor_parallel_axis)
        resolve_field("kv_head_axis", lambda: self.tensor_parallel_axis)
        resolve_field("key_sequence_axis", lambda: self.sequence_parallel_axis)
        # Default kS = S rule
        resolve_field("hidden_state_axis", lambda: self.tensor_parallel_axis)
        resolve_field("mlp_intermediate_axis", lambda: self.tensor_parallel_axis)
        resolve_field("vocab_axis", lambda: self.tensor_parallel_axis)
        resolve_field("expert_axis", lambda: self.expert_parallel_axis)

        # Resolve generation-specific axis defaults based on standard axes
        resolve_field("decode_batch_axis", lambda: get_resolved("batch_axis"))
        resolve_field("decode_head_axis", lambda: get_resolved("head_axis"))
        resolve_field("decode_kv_head_axis", lambda: get_resolved("kv_head_axis"))
        resolve_field("decode_key_sequence_axis", lambda: get_resolved("key_sequence_axis"))

        # Ensure all fields are included in resolved_values, even if not NOT_GIVEN
        for fld in dataclasses.fields(self):
            if fld.name not in resolved_values and fld.name not in [
                "_SEMANTIC_MAP",
                "_STANDARD_TO_GENERATION_ATTR_MAP",
            ]:
                resolved_values[fld.name] = getattr(self, fld.name)

        # Update the instance attributes with the resolved values
        for name, value in resolved_values.items():
            object.__setattr__(self, name, value)

        # Perform a safety check to ensure all NOT_GIVEN values were resolved
        self._safety_check()

    def _safety_check(self):
        """
        Checks if any axis attribute still has the NOT_GIVEN value after resolution.

        Raises:
            ValueError: If any attribute is still NOT_GIVEN, indicating a
                        configuration error.
        """
        for fld in dataclasses.fields(self):
            if fld.name not in ["_SEMANTIC_MAP", "_STANDARD_TO_GENERATION_ATTR_MAP"]:
                val = getattr(self, fld.name)
                if val == NOT_GIVEN:
                    raise ValueError(f"Partitioning rule `{fld.name}` was not resolved.")

    def resolve_axis(
        self,
        axes: tp.Sequence[str | None],
        mode: RUNTIME_MODE_TYPES,  # type:ignore
    ) -> list[str | None]:
        """
        Generates a Axis from a sequence of semantic axis names and a mode.

        Maps a sequence of semantic axis name strings (like BATCH, LENGTH) to the
        actual mesh axis names defined in this `PartitionAxis` instance, considering
        the current runtime mode (e.g., training vs. generation).

        Args:
            axes: A sequence of semantic axis name strings (e.g., [BATCH, LENGTH, HEAD])
                or None (or "_") for axes that shouldn't be sharded.
            mode: The current operational mode (e.g., MODE_TRAIN,
                MODE_DECODE) which determines if generation-specific
                rules should be applied.

        Returns:
            A instance representing the sharding for the given sequence of axes.

        Raises:
            ValueError: If an unknown semantic axis name is encountered or if
                a resolved axis rule is still NOT_GIVEN (should be caught
                by `_safety_check` but included for robustness).
            LookupError: If an internal attribute name derived from the semantic
                map isn't found in the instance (shouldn't happen with
                correct class definition).
        """
        resolved_rules: list[AxisType] = []

        for axis_name in axes:
            if axis_name is None or axis_name == "_":
                resolved_rules.append(None)
                continue
            if isinstance(axis_name, list):
                standard_attr_name = [self._SEMANTIC_MAP.get(axis) or axis for axis in axis_name]

            else:
                standard_attr_name = self._SEMANTIC_MAP.get(axis_name)
            if standard_attr_name is None:
                raise ValueError(f"Unknown semantic axis name: '{axis_name}'")

            target_attr_name = standard_attr_name
            if mode in GENERATION_MODES:
                gen_attr_name = self._STANDARD_TO_GENERATION_ATTR_MAP.get(standard_attr_name)
                if gen_attr_name:
                    if hasattr(self, gen_attr_name):
                        gen_val = getattr(self, gen_attr_name)
                        if gen_val is not None and gen_val is not NOT_GIVEN:
                            target_attr_name = gen_attr_name

            try:
                if isinstance(target_attr_name, list):
                    mesh_axis_rule = [getattr(self, attr_name) for attr_name in target_attr_name]
                else:
                    mesh_axis_rule: AxisType = getattr(self, target_attr_name)
            except AttributeError as e:
                raise LookupError(
                    f"Internal error: Attribute '{target_attr_name}' not found in PartitionAxis instance."
                ) from e

            if mesh_axis_rule is NOT_GIVEN:
                raise ValueError(f"Resolved axis rule for '{axis_name}' ('{target_attr_name}') is still NOT_GIVEN.")

            resolved_rules.append(mesh_axis_rule)
        return resolved_rules

    def resolve_spec(
        self,
        axes: tp.Sequence[str | None],
        mode: RUNTIME_MODE_TYPES,  # type:ignore
    ) -> PartitionSpec:
        """
        Generates a PartitionSpec from a sequence of semantic axis names and a mode.

        Maps a sequence of semantic axis name strings (like BATCH, LENGTH) to the
        actual mesh axis names defined in this `PartitionAxis` instance, considering
        the current runtime mode (e.g., training vs. generation).

        Args:
            axes: A sequence of semantic axis name strings (e.g., [BATCH, LENGTH, HEAD])
                or None (or "_") for axes that shouldn't be sharded.
            mode: The current operational mode (e.g., MODE_TRAIN,
                MODE_DECODE) which determines if generation-specific
                rules should be applied.

        Returns:
            A jax.sharding.PartitionSpec instance representing the sharding
            for the given sequence of axes.

        Raises:
            ValueError: If an unknown semantic axis name is encountered or if
                a resolved axis rule is still NOT_GIVEN (should be caught
                by `_safety_check` but included for robustness).
            LookupError: If an internal attribute name derived from the semantic
                map isn't found in the instance (shouldn't happen with
                correct class definition).
        """
        return PartitionSpec(*self.resolve_axis(axes=axes, mode=mode))

    __hash__ = hash_fn


class PartitionManager(PyTree):
    """
    Context manager for applying sharding constraints using PartitionAxis.

    This class acts as a context manager (`with PartitionManager(...)`) to
    set a context-local variable (`_CURRENT_PARTITION_MANAGER`) that makes
    the current manager implicitly available via functions like
    `get_current_partition_manager()` or the static `shard()` method.

    Args:
        paxis: The PartitionAxis instance defining the sharding strategy
               to be used within this context.
    """

    paxis: PartitionAxis

    def __post_init__(self):
        if not isinstance(self.paxis, PartitionAxis):
            raise TypeError(f"Expected PartitionAxis, got {type(self.paxis)}")

    def shard(
        self,
        x: jax.Array,
        axes: tp.Sequence[str | None] = NOT_GIVEN,
        mode: RUNTIME_MODE_TYPES | int = NOT_GIVEN,  # type:ignore
        dynamic_axes: DynamicShardingAxes | None = NOT_GIVEN,
        auto_correct: bool = True,
    ) -> jax.Array:
        """
        Applies sharding constraint to a JAX array based on the active PartitionManager context.

        Retrieves the current `PartitionManager` implicitly using `get_current_partition_manager()`
        and uses its `PartitionAxis` to resolve the semantic axis names (`axes`) into a
        `PartitionSpec`. It then applies the sharding constraint to the array `x`.

        Supports specifying axes and mode directly, or providing a `DynamicShardingAxes`
        named tuple. Can also infer the mode based on a dimension size if an integer
        `mode` is provided.

        Args:
            x: The JAX array to apply the sharding constraint to.
            axes: A sequence of semantic axis name strings or None. Required if
                `dynamic_axes` is NOT_GIVEN.
            mode: The runtime mode (string constant) or an integer representing
                the dimension index to check for mode inference. Required if
                `dynamic_axes` is NOT_GIVEN.
            dynamic_axes: An optional `DynamicShardingAxes` named tuple that
                provides both `axes` and `mode`. If provided, `axes` and
                `mode` arguments are ignored.
            auto_correct: If True, automatically corrects the resolved `PartitionSpec`
                based on array shape and mesh compatibility using
                `get_corrected_named_sharding`. Defaults to True.

        Returns:
            The array `x` with the sharding constraint applied.

        Raises:
            LookupError: If called outside of an active `PartitionManager` context.
            AssertionError: If neither `axes`/`mode` nor `dynamic_axes` are provided.
            ValueError: Propagated from `PartitionAxis.resolve_spec` or if resolved
                axis rule is NOT_GIVEN.
        """

        spec = self.resolve(
            axes=axes,
            mode=mode,
            dynamic_axes=dynamic_axes,
            shape=x.shape,
        )

        if auto_correct:
            spec = get_corrected_named_sharding(x.shape, spec).spec

        return with_sharding_constraint(x, spec)

    def resolve(
        self,
        axes: tp.Sequence[str | None] | DynamicShardingAxes = NOT_GIVEN,
        mode: RUNTIME_MODE_TYPES | int = NOT_GIVEN,  # type:ignore
        dynamic_axes: DynamicShardingAxes | None = NOT_GIVEN,
        shape: tp.Sequence[int] = NOT_GIVEN,
    ) -> PartitionSpec:
        if isinstance(axes, type) and issubclass(axes, tuple) and hasattr(axes, "_fields"):
            dynamic_axes = axes
            axes = NOT_GIVEN

        if axes is NOT_GIVEN or mode is NOT_GIVEN:
            assert dynamic_axes is not NOT_GIVEN, "if axes or mode is empty you should provide dynamic axes"
            axes = dynamic_axes.axes
            mode = dynamic_axes.mode
        if isinstance(mode, int):
            assert shape is not NOT_GIVEN, "when using dynamic mode detection shape should be provided"
            mode = MODE_DECODE if shape[mode] == 1 else MODE_TRAIN
        return self.paxis.resolve_spec(axes, mode)

    def __str__(self):
        """String representation of the PartitionManager."""
        return "PartitionManager(...)"

    def __repr__(self):
        """Representation of the PartitionManager."""
        return "PartitionManager(...)"

    __hash__ = hash_fn


def apply_logical_sharding(
    x: jax.Array,
    partition_manager: PartitionManager,
    axes: tp.Sequence[str | None] = NOT_GIVEN,
    mode: RUNTIME_MODE_TYPES | int = NOT_GIVEN,  # type:ignore
    dynamic_axes: DynamicShardingAxes | None = NOT_GIVEN,
    auto_correct: bool = True,
):
    """
    Applies logical sharding to a JAX array using an available PartitionManager.

    This function is a convenience wrapper around `PartitionManager.shard`.
    It attempts to find a `PartitionManager` from the current context first
    (`get_current_partition_manager`), and if none is found, it falls back
    to the last created manager (`get_partition_manager`).

    Args:
        x: The JAX array to apply sharding to.
        partition_manager: An explicit `PartitionManager` instance to use.
        axes: A sequence of semantic axis name strings or None. Required if
              `dynamic_axes` is NOT_GIVEN and `partition_manager` is NOT_GIVEN.
        mode: The runtime mode or dimension index for inference. Required if
              `dynamic_axes` is NOT_GIVEN and `partition_manager` is NOT_GIVEN.
        dynamic_axes: An optional `DynamicShardingAxes` tuple. If provided,
                      `axes` and `mode` are ignored.
        auto_correct: If True, automatically corrects the resolved PartitionSpec.
                      Defaults to True.

    Returns:
        The JAX array with sharding constraints applied.

    Raises:
        AssertionError: If neither `axes`/`mode` nor `dynamic_axes` are provided
                        when a manager is found or provided.
    """

    return partition_manager.shard(
        x=x,
        axes=axes,
        mode=mode,
        dynamic_axes=dynamic_axes,
        auto_correct=auto_correct,
    )
