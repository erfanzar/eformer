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

import dataclasses

import jax.numpy as jnp

from ..dtypes.precision_types import DTYPE_MAPPING, get_platform_default_half


@dataclasses.dataclass(frozen=True)
class Policy:
    """Mixed precision policy defining casting behavior."""

    param_dtype: jnp.dtype
    compute_dtype: jnp.dtype
    output_dtype: jnp.dtype

    @classmethod
    def from_string(cls, policy_str: str) -> "Policy":
        """Create policy from string like 'p=f32,c=f8_e4m3,o=f32'."""
        param_dtype = jnp.float32
        compute_dtype = output_dtype = None

        if "=" in policy_str:
            for part in policy_str.split(","):
                key, value = part.strip().split("=", 2)
                target = value.strip().lower()
                if target == "half":
                    dtype = get_platform_default_half()
                else:
                    dtype = DTYPE_MAPPING.get(target)
                if dtype is None:
                    raise ValueError(f"Unknown dtype: {value}")

                if key in ("p", "params"):
                    param_dtype = dtype
                elif key in ("c", "compute"):
                    compute_dtype = dtype
                elif key in ("o", "output"):
                    output_dtype = dtype
        else:
            target = policy_str.strip().lower()
            if target == "half":
                dtype = get_platform_default_half()
            else:
                dtype = DTYPE_MAPPING.get(target)
            if dtype is None:
                raise ValueError(f"Unknown dtype: {policy_str}")
            param_dtype = compute_dtype = output_dtype = dtype

        if compute_dtype is None:
            compute_dtype = param_dtype
        if output_dtype is None:
            output_dtype = compute_dtype

        return cls(param_dtype=param_dtype, compute_dtype=compute_dtype, output_dtype=output_dtype)
