# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import logging

import paddle
import paddle.nn as nn
import paddle.amp

from ..state import AcceleratorState
from .dataclasses import AutocastKwargs, DistributedType
from .imports import is_safetensors_available


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import load_file as safe_load_file

WEIGHTS_INDEX_NAME = "model_state.bin.index.json"


logger = logging.getLogger(__name__)


def get_mixed_precision_context_manager(native_amp: bool = False, autocast_kwargs: AutocastKwargs = None):
    """
    Return a context manager for autocasting mixed precision

    Args:
        native_amp (`bool`, *optional*, defaults to False):
            Whether mixed precision is actually enabled.
        cache_enabled (`bool`, *optional*, defaults to True):
            Whether the weight cache inside autocast should be enabled.
    """
    state = AcceleratorState()
    if autocast_kwargs is None:
        autocast_kwargs = {}
    else:
        autocast_kwargs = autocast_kwargs.to_kwargs()
    if native_amp:
        if state.mixed_precision == "fp16":
            autocast_kwargs["dtype"] = "float16"
            return paddle.amp.amp_guard(**autocast_kwargs)
        elif state.mixed_precision == "bf16" and state.distributed_type in [
            DistributedType.NO,
            DistributedType.MULTI_GPU,
        ]:
            autocast_kwargs["dtype"] = "bfloat16"
            return paddle.amp.amp_guard(**autocast_kwargs)
        else:
            return paddle.amp.amp_guard( **autocast_kwargs)
    else:
        return contextlib.nullcontext()
