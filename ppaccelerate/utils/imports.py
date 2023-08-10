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

import importlib
import sys
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata
import os
import warnings

from packaging import version

from .environment import parse_flag_from_env
from .versions import compare_versions


def _is_package_available(pkg_name):
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    if package_exists:
        try:
            _ = importlib_metadata.metadata(pkg_name)
            return True
        except importlib_metadata.PackageNotFoundError:
            return False


def is_bf16_available(ignore_tpu=False):
    "Checks if bf16 is supported, optionally ignoring the TPU"
    return True


def is_mlflow_available():
    return _is_package_available("mlflow")

def is_safetensors_available():
    return _is_package_available("safetensors")


def is_transformers_available():
    return _is_package_available("transformers")

def is_paddlenlp_available():
    return _is_package_available("paddlenlp")


def is_datasets_available():
    return _is_package_available("datasets")


def is_aim_available():
    package_exists = _is_package_available("aim")
    if package_exists:
        aim_version = version.parse(importlib_metadata.version("aim"))
        return compare_versions(aim_version, "<", "4.0.0")
    return False


def is_tensorboard_available():
    _is_tensorboard_available = True
    try:
        from torch.utils import tensorboard
    except ModuleNotFoundError:
        try:
            import tensorboardX as tensorboard
        except ModuleNotFoundError:
            _is_tensorboard_available = False
    return _is_tensorboard_available


def is_wandb_available():
    return _is_package_available("wandb")

def is_visualdl_available():
    return _is_package_available("visualdl")

def is_comet_ml_available():
    return _is_package_available("comet_ml")


def is_boto3_available():
    return _is_package_available("boto3")


def is_rich_available():
    if _is_package_available("rich"):
        if "ACCELERATE_DISABLE_RICH" in os.environ:
            warnings.warn(
                "`ACCELERATE_DISABLE_RICH` is deprecated and will be removed in v0.22.0 and deactivated by default. Please use `ACCELERATE_ENABLE_RICH` if you wish to use `rich`."
            )
            return not parse_flag_from_env("ACCELERATE_DISABLE_RICH", False)
        return parse_flag_from_env("ACCELERATE_ENABLE_RICH", False)
    return False



def is_tqdm_available():
    return _is_package_available("tqdm")


