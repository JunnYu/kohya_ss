# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import os
import sys
import tempfile

import paddle

from .state import AcceleratorState
from .utils import PrecisionType, PrepareForLaunch, patch_environment


def debug_launcher(function, args=(), num_processes=2):
    """
    Launches a training function using several processes on CPU for debugging purposes.

    <Tip warning={true}>

    This function is provided for internal testing and debugging, but it's not intended for real trainings. It will
    only use the CPU.

    </Tip>

    Args:
        function (`Callable`):
            The training function to execute.
        args (`Tuple`):
            Tuple of arguments to pass to the function (it will receive `*args`).
        num_processes (`int`, *optional*, defaults to 2):
            The number of processes to use for training.
    """

    with tempfile.NamedTemporaryFile() as tmp_file:
        # torch.distributed will expect a few environment variable to be here. We set the ones common to each
        # process here (the other ones will be set be the launcher).
        with patch_environment(
            world_size=num_processes,
            master_addr="127.0.01",
            master_port="29500",
            accelerate_mixed_precision="no",
            accelerate_debug_rdv_file=tmp_file.name,
            accelerate_use_cpu="yes",
        ):
            launcher = PrepareForLaunch(function, debug=True)
