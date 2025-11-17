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


from .mars import mars, scale_by_mars
from .utils import (
    OptaxScheduledWeightDecayState,
    create_cosine_scheduler,
    create_linear_scheduler,
    get_base_optimizer,
    optax_add_scheduled_weight_decay,
)
from .white_kron import quad, scale_by_quad, scale_by_skew, skew

__all__ = (
    "OptaxScheduledWeightDecayState",
    "create_cosine_scheduler",
    "create_linear_scheduler",
    "get_base_optimizer",
    "mars",
    "optax_add_scheduled_weight_decay",
    "quad",
    "scale_by_mars",
    "scale_by_quad",
    "scale_by_skew",
    "skew",
)
