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

from __future__ import annotations

import collections
import contextlib
import json
import os
import re
import shutil
import sys
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from types import MethodType
from typing import Any, Callable, Union

import paddle
import paddle.nn.utils
import paddle.optimizer
import paddle.amp
import paddle.io
from . import paddle_hooks as hooks

from .checkpointing import load_accelerator_state, load_custom_state, save_accelerator_state, save_custom_state
from .logging import get_logger
from .optimizer import AcceleratedOptimizer
from .scheduler import AcceleratedScheduler
from .state import AcceleratorState, GradientState, PartialState
from .tracking import LOGGER_TYPE_TO_CLASS, GeneralTracker, filter_trackers
from .utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    AutocastKwargs,
    DistributedDataParallelKwargs,
    DistributedType,
    GradientAccumulationPlugin,
    GradScalerKwargs,
    InitProcessGroupKwargs,
    KwargsHandler,
    LoggerType,
    PrecisionType,
    ProjectConfiguration,
    compare_versions,
    convert_outputs_to_fp32,
    extract_model_from_parallel,
    gather,
    get_mixed_precision_context_manager,
    get_pretty_name,
    is_bf16_available,
    is_safetensors_available,
    is_torch_version,
    pad_across_processes,
    parse_choice_from_env,
    recursively_apply,
    reduce,
    release_memory,
    save,
    wait_for_everyone,
)

from paddle.optimizer.lr import LRScheduler

logger = get_logger(__name__)


class Accelerator:
    """
    Creates an instance of an accelerator for distributed training (on multi-GPU, TPU) or mixed precision training.

    Args:
        device_placement (`bool`, *optional*, defaults to `True`):
            Whether or not the accelerator should put objects on device (tensors yielded by the dataloader, model,
            etc...).
        split_batches (`bool`, *optional*, defaults to `False`):
            Whether or not the accelerator should split the batches yielded by the dataloaders across the devices. If
            `True` the actual batch size used will be the same on any kind of distributed processes, but it must be a
            round multiple of the `num_processes` you are using. If `False`, actual batch size used will be the one set
            in your script multiplied by the number of processes.
        mixed_precision (`str`, *optional*):
            Whether or not to use mixed precision training. Choose from 'no','fp16','bf16 or 'fp8'. Will default to the
            value in the environment variable `ACCELERATE_MIXED_PRECISION`, which will use the default value in the
            accelerate config of the current system or the flag passed with the `accelerate.launch` command. 'fp8'
            requires the installation of transformers-engine.
        gradient_accumulation_steps (`int`, *optional*, default to 1):
            The number of steps that should pass before gradients are accumulated. A number > 1 should be combined with
            `Accelerator.accumulate`. If not passed, will default to the value in the environment variable
            `ACCELERATE_GRADIENT_ACCUMULATION_STEPS`. Can also be configured through a `GradientAccumulationPlugin`.
        cpu (`bool`, *optional*):
            Whether or not to force the script to execute on CPU. Will ignore GPU available if set to `True` and force
            the execution on one process only.
        deepspeed_plugin (`DeepSpeedPlugin`, *optional*):
            Tweak your DeepSpeed related args using this argument. This argument is optional and can be configured
            directly using *accelerate config*
        fsdp_plugin (`FullyShardedDataParallelPlugin`, *optional*):
            Tweak your FSDP related args using this argument. This argument is optional and can be configured directly
            using *accelerate config*
        megatron_lm_plugin (`MegatronLMPlugin`, *optional*):
            Tweak your MegatronLM related args using this argument. This argument is optional and can be configured
            directly using *accelerate config*
        rng_types (list of `str` or [`~utils.RNGType`]):
            The list of random number generators to synchronize at the beginning of each iteration in your prepared
            dataloaders. Should be one or several of:

            - `"torch"`: the base torch random number generator
            - `"cuda"`: the CUDA random number generator (GPU only)
            - `"xla"`: the XLA random number generator (TPU only)
            - `"generator"`: the `torch.Generator` of the sampler (or batch sampler if there is no sampler in your
              dataloader) or of the iterable dataset (if it exists) if the underlying dataset is of that type.

            Will default to `["torch"]` for PyTorch versions <=1.5.1 and `["generator"]` for PyTorch versions >= 1.6.
        log_with (list of `str`, [`~utils.LoggerType`] or [`~tracking.GeneralTracker`], *optional*):
            A list of loggers to be setup for experiment tracking. Should be one or several of:

            - `"all"`
            - `"tensorboard"`
            - `"wandb"`
            - `"comet_ml"`
            If `"all"` is selected, will pick up all available trackers in the environment and initialize them. Can
            also accept implementations of `GeneralTracker` for custom trackers, and can be combined with `"all"`.
        project_config (`ProjectConfiguration`, *optional*):
            A configuration for how saving the state can be handled.
        project_dir (`str`, `os.PathLike`, *optional*):
            A path to a directory for storing data such as logs of locally-compatible loggers and potentially saved
            checkpoints.
        dispatch_batches (`bool`, *optional*):
            If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process
            and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose
            underlying dataset is an `IterableDataset`, `False` otherwise.
        even_batches (`bool`, *optional*, defaults to `True`):
            If set to `True`, in cases where the total batch size across all processes does not exactly divide the
            dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among
            all workers.
        step_scheduler_with_optimizer (`bool`, *optional`, defaults to `True`):
            Set `True` if the learning rate scheduler is stepped at the same time as the optimizer, `False` if only
            done under certain circumstances (at the end of each epoch, for instance).
        kwargs_handlers (`list[KwargHandler]`, *optional*)
            A list of `KwargHandler` to customize how the objects related to distributed training or mixed precision
            are created. See [kwargs](kwargs) for more information.
        dynamo_backend (`str` or `DynamoBackend`, *optional*, defaults to `"no"`):
            Set to one of the possible dynamo backends to optimize your training with torch dynamo.
        gradient_accumulation_plugin (`GradientAccumulationPlugin`, *optional*):
            A configuration for how gradient accumulation should be handled, if more tweaking than just the
            `gradient_accumulation_steps` is needed.

    **Available attributes:**

        - **device** (`str`) -- The device to use.
        - **distributed_type** ([`~utils.DistributedType`]) -- The distributed training configuration.
        - **local_process_index** (`int`) -- The process index on the current machine.
        - **mixed_precision** (`str`) -- The configured mixed precision mode.
        - **num_processes** (`int`) -- The total number of processes used for training.
        - **optimizer_step_was_skipped** (`bool`) -- Whether or not the optimizer update was skipped (because of
          gradient overflow in mixed precision), in which
        case the learning rate should not be changed.
        - **process_index** (`int`) -- The overall index of the current process among all processes.
        - **state** ([`~state.AcceleratorState`]) -- The distributed setup state.
        - **sync_gradients** (`bool`) -- Whether the gradients are currently being synced across all processes.
        - **use_distributed** (`bool`) -- Whether the current configuration is for distributed training.
    """

    def __init__(
        self,
        device_placement: bool = True,
        split_batches: bool = False,
        mixed_precision: PrecisionType | str | None = None,
        gradient_accumulation_steps: int = 1,
        # cpu: bool = False,
        # rng_types: list[str | RNGType] | None = None,
        log_with: str | LoggerType | GeneralTracker | list[str | LoggerType | GeneralTracker] | None = None,
        project_dir: str | os.PathLike | None = None,
        project_config: ProjectConfiguration | None = None,
        gradient_accumulation_plugin: GradientAccumulationPlugin | None = None,
        # dispatch_batches: bool | None = None,
        # even_batches: bool = True,
        step_scheduler_with_optimizer: bool = True,
        kwargs_handlers: list[KwargsHandler] | None = None,
    ):
        if project_config is not None:
            self.project_configuration = project_config
        else:
            self.project_configuration = ProjectConfiguration(project_dir=project_dir)
        if project_dir is not None and self.project_dir is None:
            self.project_configuration.set_directories(project_dir)
        if mixed_precision is not None:
            mixed_precision = str(mixed_precision)
            if mixed_precision not in PrecisionType:
                raise ValueError(
                    f"Unknown mixed_precision mode: {mixed_precision}. Choose between {PrecisionType.list()}"
                )

        # Kwargs handlers
        self.ddp_handler = None
        self.scaler_handler = None
        self.init_handler = None
        self.autocast_handler = None
        if kwargs_handlers is not None:
            for handler in kwargs_handlers:
                assert isinstance(
                    handler, KwargsHandler
                ), f"Unsupported kwargs handler passed: {handler}, must be one that inherits `accelerate.utils.KwargsHandler`."
                if isinstance(handler, DistributedDataParallelKwargs):
                    if self.ddp_handler is not None:
                        raise ValueError("You can only pass one `DistributedDataParallelKwargs` in `kwargs_handler`.")
                    else:
                        self.ddp_handler = handler
                elif isinstance(handler, GradScalerKwargs):
                    if self.scaler_handler is not None:
                        raise ValueError("You can only pass one `GradScalerKwargs` in `kwargs_handler`.")
                    else:
                        self.scaler_handler = handler
                elif isinstance(handler, InitProcessGroupKwargs):
                    if self.init_handler is not None:
                        raise ValueError("You can only pass one `InitProcessGroupKwargs` in `kwargs_handler`.")
                    else:
                        self.init_handler = handler
                elif isinstance(handler, AutocastKwargs):
                    if self.autocast_handler is not None:
                        raise ValueError("You can only pass one `AutocastKwargs` in `kwargs_handler`.")
                    else:
                        self.autocast_handler = handler

        kwargs = self.init_handler.to_kwargs() if self.init_handler is not None else {}
        self.state = AcceleratorState(
            mixed_precision=mixed_precision,
            cpu=False,
            _from_accelerator=True,
            **kwargs,
        )

        trackers = filter_trackers(log_with, self.logging_dir)
        if len(trackers) < 1 and log_with is not None:
            warnings.warn(f"`log_with={log_with}` was passed but no supported trackers are currently installed.")
        self.log_with = trackers

        if gradient_accumulation_plugin is not None:
            if gradient_accumulation_steps != 1:
                raise ValueError(
                    "You can only pass one of `gradient_accumulation_steps` and `gradient_accumulation_plugin`. Please only pass in the created `GradientAccumulationPlugin` object."
                )
        else:
            gradient_accumulation_steps = int(
                parse_choice_from_env("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", gradient_accumulation_steps)
            )
            gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=gradient_accumulation_steps)
        self.gradient_state = GradientState(
            gradient_accumulation_plugin=gradient_accumulation_plugin,
        )

        self.device_placement = device_placement
        self.split_batches = split_batches
        # self.dispatch_batches = dispatch_batches
        # self.even_batches = even_batches
        self.step_scheduler_with_optimizer = step_scheduler_with_optimizer

        # Mixed precision attributes
        self.scaler = None
        self.native_amp = False
        if self.state.mixed_precision == "fp16":
            self.native_amp = True
            kwargs = self.scaler_handler.to_kwargs() if self.scaler_handler is not None else {}
            self.scaler = paddle.amp.GradScaler(**kwargs)

        elif self.state.mixed_precision == "bf16":
            self.native_amp = True

        # Start of internal step tracking
        self.step = 0

        # Internal references to the training objects
        self._optimizers = []
        self._models = []
        self._schedulers = []
        self._dataloaders = []
        self._custom_objects = []

        # Hooks
        self._load_model_state_pre_hook = OrderedDict()
        self._save_model_state_pre_hook = OrderedDict()

        # RNG Types
        # self.rng_types = rng_types
        # if self.rng_types is None:
        #     self.rng_types = ["generator"]

    @property
    def use_distributed(self):
        """
        Whether the Accelerator is configured for distributed training
        """
        return self.state.use_distributed

    @property
    def distributed_type(self):
        return self.state.distributed_type

    @property
    def num_processes(self):
        return self.state.num_processes

    @property
    def process_index(self):
        return self.state.process_index

    @property
    def local_process_index(self):
        return self.state.local_process_index

    @property
    def device(self):
        return self.state.device

    @property
    def project_dir(self):
        return self.project_configuration.project_dir

    @property
    def logging_dir(self):
        return self.project_configuration.logging_dir

    @property
    def save_iteration(self):
        return self.project_configuration.iteration

    @property
    def is_main_process(self):
        """True for one process only."""
        return self.state.is_main_process

    @property
    def is_local_main_process(self):
        """True for one process per server."""
        return self.state.is_local_main_process

    @property
    def use_fp16(self):
        warnings.warn(
            "The `use_fp16` property is deprecated and will be removed in version 1.0 of Accelerate use "
            "`Accelerator.mixed_precision == 'fp16'` instead.",
            FutureWarning,
        )
        return self.mixed_precision != "no"

    @property
    def is_last_process(self):
        return self.process_index == self.num_processes - 1

    @property
    def mixed_precision(self):
        return self.state.mixed_precision

    @contextmanager
    def split_between_processes(self, inputs: list | tuple | dict | paddle.Tensor, apply_padding: bool = False):
        """
        Splits `input` between `self.num_processes` quickly and can be then used on that process. Useful when doing
        distributed inference, such as with different prompts.

        Note that when using a `dict`, all keys need to have the same number of elements.

        Args:
            inputs (`list`, `tuple`, `paddle.Tensor`, or `dict` of `list`/`tuple`/`paddle.Tensor`):
                The input to split between processes.
            apply_padding (`bool`, `optional`, defaults to `False`):
                Whether to apply padding by repeating the last element of the input so that all processes have the same
                number of elements. Useful when trying to perform actions such as `Accelerator.gather()` on the outputs
                or passing in less inputs than there are processes. If so, just remember to drop the padded elements
                afterwards.

        Example:

        ```python
        # Assume there are two processes
        from accelerate import Accelerator

        accelerator = Accelerator()
        with accelerator.split_between_processes(["A", "B", "C"]) as inputs:
            print(inputs)
        # Process 0
        ["A", "B"]
        # Process 1
        ["C"]

        with accelerator.split_between_processes(["A", "B", "C"], apply_padding=True) as inputs:
            print(inputs)
        # Process 0
        ["A", "B"]
        # Process 1
        ["C", "C"]
        ```
        """
        with PartialState().split_between_processes(inputs, apply_padding=apply_padding) as inputs:
            yield inputs

    def on_main_process(self, function: Callable[..., Any] = None):
        """
        A decorator that will run the decorated function on the main process only. Can also be called using the
        `PartialState` class.

        Args:
            function (`Callable`): The function to decorate.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()


        >>> @accelerator.on_main_process
        ... def print_something():
        ...     print("This will be printed by process 0 only.")


        >>> print_something()
        "This will be printed by process 0 only"
        ```
        """
        # For times when the `Accelerator` object itself utilizes this decorator.
        if function is None:
            if "Accelerator." in self.__qualname__:
                function = self
            else:
                raise ValueError(
                    "The `on_main_process` decorator must be called with a function on an instantiated `Accelerator` object."
                )

        def _inner(*args, **kwargs):
            return PartialState().on_main_process(function)(*args, **kwargs)

        return _inner

    def on_local_main_process(self, function: Callable[..., Any] = None):
        """
        A decorator that will run the decorated function on the local main process only. Can also be called using the
        `PartialState` class.

        Args:
            function (`Callable`): The function to decorate.

        Example:
        ```python
        # Assume we have 2 servers with 4 processes each.
        from accelerate import Accelerator

        accelerator = Accelerator()


        @accelerator.on_local_main_process
        def print_something():
            print("This will be printed by process 0 only on each server.")


        print_something()
        # On server 1:
        "This will be printed by process 0 only"
        # On server 2:
        "This will be printed by process 0 only"
        ```
        """
        # For times when the `Accelerator` object itself utilizes this decorator.
        if function is None:
            if "Accelerator." in self.__qualname__:
                function = self
            else:
                raise ValueError(
                    "The `on_local_main_process` decorator must be called with a function on an instantiated `Accelerator` object."
                )

        def _inner(*args, **kwargs):
            return PartialState().on_local_main_process(function)(*args, **kwargs)

        return _inner

    def on_last_process(self, function: Callable[..., Any]):
        """
        A decorator that will run the decorated function on the last process only. Can also be called using the
        `PartialState` class.

        Args:
            function (`Callable`): The function to decorate.

        Example:
        ```python
        # Assume we have 4 processes.
        from accelerate import Accelerator

        accelerator = Accelerator()


        @accelerator.on_last_process
        def print_something():
            print(f"Printed on process {accelerator.process_index}")


        print_something()
        "Printed on process 3"
        ```
        """
        # For times when the `Accelerator` object itself utilizes this decorator.
        if function is None:
            if "Accelerator." in self.__qualname__:
                function = self
            else:
                raise ValueError(
                    "The `on_last_process` decorator must be called with a function on an instantiated `Accelerator` object."
                )

        def _inner(*args, **kwargs):
            return PartialState().on_last_process(function)(*args, **kwargs)

        return _inner

    def on_process(self, function: Callable[..., Any] = None, process_index: int = None):
        """
        A decorator that will run the decorated function on a given process index only. Can also be called using the
        `PartialState` class.

        Args:
            function (`Callable`, `optional`):
                The function to decorate.
            process_index (`int`, `optional`):
                The index of the process on which to run the function.

        Example:
        ```python
        # Assume we have 4 processes.
        from accelerate import Accelerator

        accelerator = Accelerator()


        @accelerator.on_process(process_index=2)
        def print_something():
            print(f"Printed on process {accelerator.process_index}")


        print_something()
        "Printed on process 2"
        ```
        """
        # Initial construction of the decorator.
        if (self is not None) and (process_index is not None) and (function is None):
            return partial(self.on_process, process_index=process_index)
        # For times when the `Accelerator` object itself utilizes this decorator.
        if function is None:
            if "Accelerator." in self.__qualname__:
                function = self
            else:
                raise ValueError(
                    "The `on_main_process` decorator must be called with a function on an instantiated `Accelerator` object."
                )

        def _inner(*args, **kwargs):
            return PartialState().on_process(function, process_index)(*args, **kwargs)

        return _inner

    def on_local_process(self, function: Callable[..., Any] = None, local_process_index: int = None):
        """
        A decorator that will run the decorated function on a given local process index only. Can also be called using
        the `PartialState` class.

        Args:
            function (`Callable`, *optional*):
                The function to decorate.
            local_process_index (`int`, *optional*):
                The index of the local process on which to run the function.

        Example:
        ```python
        # Assume we have 2 servers with 4 processes each.
        from accelerate import Accelerator

        accelerator = Accelerator()


        @accelerator.on_local_process(local_process_index=2)
        def print_something():
            print(f"Printed on process {accelerator.local_process_index}")


        print_something()
        # On server 1:
        "Printed on process 2"
        # On server 2:
        "Printed on process 2"
        ```
        """
        # Initial construction of the decorator.
        if (self is not None) and (local_process_index is not None) and (function is None):
            return partial(self.on_local_process, local_process_index=local_process_index)
        # For times when the `Accelerator` object itself utilizes this decorator.
        if function is None:
            if "Accelerator." in self.__qualname__:
                function = self
            else:
                raise ValueError(
                    "The `on_main_process` decorator must be called with a function on an instantiated `Accelerator` object."
                )

        def _inner(*args, **kwargs):
            return PartialState().on_local_process(function, local_process_index)(*args, **kwargs)

        return _inner

    @contextmanager
    def main_process_first(self):
        """
        Lets the main process go first inside a with block.

        The other processes will enter the with block after the main process exits.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> with accelerator.main_process_first():
        ...     # This will be printed first by process 0 then in a seemingly
        ...     # random order by the other processes.
        ...     print(f"This will be printed by process {accelerator.process_index}")
        ```
        """
        with self.state.main_process_first():
            yield

    @contextmanager
    def local_main_process_first(self):
        """
        Lets the local main process go inside a with block.

        The other processes will enter the with block after the main process exits.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> with accelerator.local_main_process_first():
        ...     # This will be printed first by local process 0 then in a seemingly
        ...     # random order by the other processes.
        ...     print(f"This will be printed by process {accelerator.local_process_index}")
        ```
        """
        with self.state.local_main_process_first():
            yield

    @contextmanager
    def no_sync(self, model):
        """
        A context manager to disable gradient synchronizations across DDP processes by calling
        `torch.nn.parallel.DistributedDataParallel.no_sync`.

        If `model` is not in DDP, this context manager does nothing

        Args:
            model (`paddle.nn.Layer`):
                PyTorch Module that was prepared with `Accelerator.prepare`

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> dataloader, model, optimizer = accelerator.prepare(dataloader, model, optimizer)
        >>> input_a = next(iter(dataloader))
        >>> input_b = next(iter(dataloader))

        >>> with accelerator.no_sync():
        ...     outputs = model(input_a)
        ...     loss = loss_func(outputs)
        ...     accelerator.backward(loss)
        ...     # No synchronization across processes, only accumulate gradients
        >>> outputs = model(input_b)
        >>> accelerator.backward(loss)
        >>> # Synchronization across all processes
        >>> optimizer.step()
        >>> optimizer.zero_grad()
        ```
        """
        context = contextlib.nullcontext
        if self.use_distributed:
            context = getattr(model, "no_sync", context)

        with context():
            yield

    @staticmethod
    @contextmanager
    def trigger_sync_in_backward(model):
        yield
        # """Trigger the sync of the gradients in the next backward pass of the model after multiple forward passes under
        # `Accelerator.no_sync` (only applicable in multi-GPU scenarios).

        #         If the script is not launched in distributed mode, this context manager does nothing.

        #         Args:
        #             model (`paddle.nn.Layer`):
        #                 The model for which to trigger the gradient synchronization.

        #         Example:

        #         ```python
        #         >>> from accelerate import Accelerator

        #         >>> accelerator = Accelerator()
        #         >>> dataloader, model, optimizer = accelerator.prepare(dataloader, model, optimizer)

        #         >>> with accelerator.no_sync():
        #         ...     loss_a = loss_func(model(input_a))  # first forward pass
        #         ...     loss_b = loss_func(model(input_b))  # second forward pass
        #         >>> accelerator.backward(loss_a)  # No synchronization across processes, only accumulate gradients
        #         >>> with accelerator.trigger_sync_in_backward(model):
        #         ...     accelerator.backward(loss_b)  # Synchronization across all processes
        #         >>> optimizer.step()
        #         >>> optimizer.zero_grad()
        #         ```
        # """
        # if not isinstance(model, paddle.DataParallel):
        #     yield
        #     return

        # old_require_backward_grad_sync = model.require_backward_grad_sync
        # old_require_forward_param_sync = model.require_forward_param_sync

        # # EXPERIMENTAL: This will force grad sync during `backward()`, but it is unknown if it breaks other DDP features.
        # # https://github.com/pytorch/pytorch/blob/e1502c0cdbfd17548c612f25d5a65b1e4b86224d/torch/nn/parallel/distributed.py#L1453-L1466
        # model.require_backward_grad_sync = True
        # model.require_forward_param_sync = True
        # # https://github.com/pytorch/pytorch/blob/e1502c0cdbfd17548c612f25d5a65b1e4b86224d/torch/csrc/distributed/c10d/reducer.cpp#L1371-L1402
        # model.reducer.prepare_for_backward([])
        # try:
        #     yield
        # finally:
        #     model.require_backward_grad_sync = old_require_backward_grad_sync
        #     model.require_forward_param_sync = old_require_forward_param_sync

    def _do_sync(self):
        "Sets the right `sync_gradients` context and either resets or increases `self.step`"
        if self.gradient_state.sync_with_dataloader and self.gradient_state.end_of_dataloader:
            self.step = 0
            self.gradient_state._set_sync_gradients(True)
        else:
            self.step += 1
            self.gradient_state._set_sync_gradients((self.step % self.gradient_state.num_steps) == 0)

    @property
    def sync_gradients(self):
        return self.gradient_state.sync_gradients

    @sync_gradients.setter
    def sync_gradients(self, sync_gradients):
        self.gradient_state.sync_gradients = sync_gradients

    @property
    def gradient_accumulation_steps(self):
        return self.gradient_state.num_steps

    @gradient_accumulation_steps.setter
    def gradient_accumulation_steps(self, gradient_accumulation_steps):
        self.gradient_state.plugin_kwargs.update({"num_steps": gradient_accumulation_steps})

    @contextmanager
    def accumulate(self, *models):
        """
        A context manager that will lightly wrap around and perform gradient accumulation automatically

        Args:
            *models (list of `paddle.nn.Layer`):
                PyTorch Modules that was prepared with `Accelerator.prepare`. Models passed to `accumulate()` will skip
                gradient syncing during backward pass in distributed training

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(gradient_accumulation_steps=1)
        >>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

        >>> for input, output in dataloader:
        ...     with accelerator.accumulate(model):
        ...         outputs = model(input)
        ...         loss = loss_func(outputs)
        ...         loss.backward()
        ...         optimizer.step()
        ...         scheduler.step()
        ...         optimizer.zero_grad()
        ```
        """
        self._do_sync()
        with contextlib.ExitStack() as cm_stack:
            for m in models:
                cm_stack.enter_context(contextlib.nullcontext() if self.sync_gradients else self.no_sync(m))
            yield

    @contextmanager
    def join_uneven_inputs(self, joinables, even_batches=None):
        yield

    def print(self, *args, **kwargs):
        """
        Drop in replacement of `print()` to only print once per server.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> accelerator.print("Hello world!")
        ```
        """
        self.state.print(*args, **kwargs)

    def _prepare_one(self, obj, first_pass=False, device_placement=None):
        # First pass of preparation: DataLoader, model, optimizer
        if first_pass:
            if isinstance(obj, paddle.io.DataLoader):
                return self.prepare_data_loader(obj, device_placement=device_placement)
            elif isinstance(obj, paddle.nn.Layer):
                return self.prepare_model(obj, device_placement=device_placement)
            elif isinstance(obj, paddle.optimizer.Optimizer):
                optimizer = self.prepare_optimizer(obj, device_placement=device_placement)
                return optimizer
        # Second pass of preparation: LR scheduler (which need the full list of optimizers)
        elif isinstance(obj, LRScheduler):
            scheduler = self.prepare_scheduler(obj)
            return scheduler
        # Return the unprocessed object if previous criteria was not met
        return obj

    def prepare(self, *args, device_placement=None):
        """
        Prepare all objects passed in `args` for distributed training and mixed precision, then return them in the same
        order.

        Args:
            *args (list of objects):
                Any of the following type of objects:

                - `paddle.io.DataLoader`: PyTorch Dataloader
                - `paddle.nn.Layer`: PyTorch Module
                - `paddle.optimizer.Optimizer`: PyTorch Optimizer
                - `torch.optim.lr_scheduler.LRScheduler`: PyTorch LR Scheduler

            device_placement (`list[bool]`, *optional*):
                Used to customize whether automatic device placement should be performed for each object passed. Needs
                to be a list of the same length as `args`. Not compatible with DeepSpeed or FSDP.

        <Tip>

          You don't need to prepare a model if you only use it for inference without any kind of mixed precision

        </Tip>

        Examples:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> # Assume a model, optimizer, data_loader and scheduler are defined
        >>> model, optimizer, data_loader, scheduler = accelerator.prepare(model, optimizer, data_loader, scheduler)
        ```

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> # Assume a model, optimizer, data_loader and scheduler are defined
        >>> device_placement = [True, True, False, False]
        >>> # Will place the first to items passed in automatically to the right device but not the last two.
        >>> model, optimizer, data_loader, scheduler = accelerator.prepare(
        ...     model, optimizer, data_loader, scheduler, device_placement=device_placement
        ... )
        ```
        """
        if device_placement is None:
            device_placement = [None for _ in args]
        elif len(device_placement) != len(args):
            raise ValueError(
                f"`device_placement` should be a list with {len(args)} elements (the number of objects passed)."
            )

        for obj in args:
            if (
                isinstance(obj, paddle.nn.Layer)
                and self.verify_device_map(obj)
                and self.distributed_type != DistributedType.NO
            ):
                raise ValueError(
                    "You can't train a model that has been loaded with `device_map='auto'` in any distributed mode."
                    " Please rerun your script specifying `--num_processes=1` or by launching with `python {{myscript.py}}`."
                )


        result = tuple(
            self._prepare_one(obj, first_pass=True, device_placement=d) for obj, d in zip(args, device_placement)
        )
        result = tuple(self._prepare_one(obj, device_placement=d) for obj, d in zip(result, device_placement))

     
        for item in result:
            if any(
                item in container
                for container in (self._dataloaders, self._models, self._optimizers, self._schedulers)
            ):
                setattr(item, "_is_accelerate_prepared", True)

        return result if len(result) > 1 else result[0]

    def prepare_model(self, model: paddle.nn.Layer, device_placement: bool = None, evaluation_mode: bool = False):
        """
        Prepares a PyTorch model for training in any distributed setup. It is recommended to use
        [`Accelerator.prepare`] instead.

        Args:
            model (`paddle.nn.Layer`):
                A PyTorch model to prepare. You don't need to prepare a model if it is used only for inference without
                any kind of mixed precision
            device_placement (`bool`, *optional*):
                Whether or not to place the model on the proper device. Will default to `self.device_placement`.
            evaluation_mode (`bool`, *optional*, defaults to `False`):
                Whether or not to set the model for evaluation only, by just applying mixed precision and
                `torch.compile` (if configured in the `Accelerator` object).

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> # Assume a model is defined
        >>> model = accelerator.prepare_model(model)
        ```
        """
        if device_placement is None:
            device_placement = self.device_placement
        self._models.append(model)

        if self.verify_device_map(model) and self.distributed_type != DistributedType.NO:
            raise ValueError(
                "You can't train a model that has been loaded with `device_map='auto'` in any distributed mode."
                " Please rerun your script specifying `--num_processes=1` or by launching with `python {{myscript.py}}`."
            )

        elif device_placement and not self.verify_device_map(model):
            model = model.to(self.device)

        if self.native_amp:
            model._original_forward = model.forward
            model_forward_func = model.forward.__func__ if hasattr(model.forward, "__func__") else model.forward
            autocast_context = get_mixed_precision_context_manager(self.native_amp, self.autocast_handler)
            new_forward = autocast_context(model_forward_func)
            if hasattr(model.forward, "__func__"):
                model.forward = MethodType(new_forward, model)
                if hasattr(model.forward, "__func__"):
                    model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)
                else:
                    model.forward = MethodType(convert_outputs_to_fp32(new_forward), model)
            else:
                model.forward = MethodType(convert_outputs_to_fp32(new_forward), model)

        if not evaluation_mode:
            if self.distributed_type in (
                DistributedType.MULTI_GPU,
            ):
                if any(not p.stop_gradient for p in model.parameters()):
                    kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
                    # TODO kwargs
                    model = paddle.DataParallel(
                        model,  
                    )

        return model

    def prepare_data_loader(self, data_loader: paddle.io.DataLoader, device_placement=None):
        """
        Prepares a PyTorch DataLoader for training in any distributed setup. It is recommended to use
        [`Accelerator.prepare`] instead.

        Args:
            data_loader (`paddle.io.DataLoader`):
                A vanilla PyTorch DataLoader to prepare
            device_placement (`bool`, *optional*):
                Whether or not to place the batches on the proper device in the prepared dataloader. Will default to
                `self.device_placement`.

        Example:

        ```python
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> data_loader = paddle.io.DataLoader(...)
        >>> data_loader = accelerator.prepare_data_loader(data_loader, device_placement=True)
        ```
        """
        # Ensure we can't double wrap a DataLoader due to `find_batch_size`
        # if getattr(data_loader, "_is_accelerate_prepared", False):
        #     if data_loader not in self._dataloaders:
        #         self._dataloaders.append(data_loader)
        #     return data_loader
        # if device_placement is None:
        #     device_placement = self.device_placement if self.distributed_type != DistributedType.TPU else False
        # prepared_data_loader = prepare_data_loader(
        #     data_loader,
        #     self.device,
        #     num_processes=self.num_processes,
        #     process_index=self.process_index,
        #     split_batches=self.split_batches,
        #     put_on_device=device_placement,
        #     rng_types=self.rng_types.copy(),
        #     dispatch_batches=self.dispatch_batches,
        #     even_batches=self.even_batches,
        # )
        self._dataloaders.append(data_loader)
        return data_loader

    def prepare_optimizer(self, optimizer: paddle.optimizer.Optimizer, device_placement=None):
        """
        Prepares a PyTorch Optimizer for training in any distributed setup. It is recommended to use
        [`Accelerator.prepare`] instead.

        Args:
            optimizer (`paddle.optimizer.Optimizer`):
                A vanilla PyTorch optimizer to prepare
            device_placement (`bool`, *optional*):
                Whether or not to place the optimizer on the proper device. Will default to `self.device_placement`.

        Example:

        ```python
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> optimizer = torch.optim.Adam(...)
        >>> optimizer = accelerator.prepare_optimizer(optimizer, device_placement=True)
        ```
        """
        # Ensure we can't double wrap an optimizer due to `find_batch_size`
        if getattr(optimizer, "_is_accelerate_prepared", False):
            if optimizer not in self._optimizers:
                self._optimizers.append(optimizer)
            return optimizer
        if device_placement is None:
            device_placement = self.device_placement
        optimizer = AcceleratedOptimizer(optimizer, device_placement=device_placement, scaler=self.scaler)
        self._optimizers.append(optimizer)
        return optimizer

    def prepare_scheduler(self, scheduler: LRScheduler):
        """
        Prepares a PyTorch Scheduler for training in any distributed setup. It is recommended to use
        [`Accelerator.prepare`] instead.

        Args:
            scheduler (`torch.optim.lr_scheduler.LRScheduler`):
                A vanilla PyTorch scheduler to prepare

        Example:

        ```python
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> optimizer = torch.optim.Adam(...)
        >>> scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, ...)
        >>> scheduler = accelerator.prepare_scheduler(scheduler)
        ```
        """
        # Ensure we can't double wrap a scheduler due to `find_batch_size`
        if getattr(scheduler, "_is_accelerate_prepared", False):
            if scheduler not in self._schedulers:
                self._schedulers.append(scheduler)
            return scheduler
        # We try to find the optimizer associated with `scheduler`, the default is the full list.
        optimizer = self._optimizers
        for opt in self._optimizers:
            if getattr(scheduler, "optimizer", None) == opt.optimizer:
                optimizer = opt
                break
        scheduler = AcceleratedScheduler(
            scheduler,
            optimizer,
            step_with_optimizer=self.step_scheduler_with_optimizer,
            split_batches=self.split_batches,
        )
        self._schedulers.append(scheduler)
        return scheduler

    def backward(self, loss, **kwargs):
        """
        Scales the gradients in accordance to the `GradientAccumulationPlugin` and calls the correct `backward()` based
        on the configuration.

        Should be used in lieu of `loss.backward()`.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(gradient_accumulation_steps=2)
        >>> outputs = model(inputs)
        >>> loss = loss_fn(outputs, labels)
        >>> accelerator.backward(loss)
        ```
        """
        loss = loss / self.gradient_accumulation_steps
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def unscale_gradients(self, optimizer=None):
        """
        Unscale the gradients in mixed precision training with AMP. This is a noop in all other settings.

        Likely should be called through [`Accelerator.clip_grad_norm_`] or [`Accelerator.clip_grad_value_`]

        Args:
            optimizer (`paddle.optimizer.Optimizer` or `list[paddle.optimizer.Optimizer]`, *optional*):
                The optimizer(s) for which to unscale gradients. If not set, will unscale gradients on all optimizers
                that were passed to [`~Accelerator.prepare`].

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> model, optimizer = accelerator.prepare(model, optimizer)
        >>> outputs = model(inputs)
        >>> loss = loss_fn(outputs, labels)
        >>> accelerator.backward(loss)
        >>> accelerator.unscale_gradients(optimizer=optimizer)
        ```
        """
        if self.native_amp and self.mixed_precision == "fp16":
            if optimizer is None:
                # TODO: this unscales all optimizers where we should only unscale the one where parameters are.
                optimizer = self._optimizers
            elif not isinstance(optimizer, (tuple, list)):
                optimizer = [optimizer]
            for opt in optimizer:
                while isinstance(opt, AcceleratedOptimizer):
                    opt = opt.optimizer
                self.scaler.unscale_(opt)

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        """
        Should be used in place of `torch.nn.utils.clip_grad_norm_`.

        Returns:
            `paddle.Tensor`: Total norm of the parameter gradients (viewed as a single vector).

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(gradient_accumulation_steps=2)
        >>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

        >>> for input, target in dataloader:
        ...     optimizer.zero_grad()
        ...     output = model(input)
        ...     loss = loss_func(output, target)
        ...     accelerator.backward(loss)
        ...     if accelerator.sync_gradients:
        ...         accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        ...     optimizer.step()
        ```
        """
        self.unscale_gradients()
        return paddle.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)

    def clip_grad_value_(self, parameters, clip_value):
        """
        Should be used in place of `torch.nn.utils.clip_grad_value_`.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(gradient_accumulation_steps=2)
        >>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

        >>> for input, target in dataloader:
        ...     optimizer.zero_grad()
        ...     output = model(input)
        ...     loss = loss_func(output, target)
        ...     accelerator.backward(loss)
        ...     if accelerator.sync_gradients:
        ...         accelerator.clip_grad_value_(model.parameters(), clip_value)
        ...     optimizer.step()
        ```
        """
        self.unscale_gradients()
        paddle.nn.utils.clip_grad_value_(parameters, clip_value)

    def gather(self, tensor):
        """
        Gather the values in *tensor* across all processes and concatenate them on the first dimension. Useful to
        regroup the predictions from all processes when doing evaluation.

        Note:
            This gather happens in all processes.

        Args:
            tensor (`paddle.Tensor`, or a nested tuple/list/dictionary of `paddle.Tensor`):
                The tensors to gather across all processes.

        Returns:
            `paddle.Tensor`, or a nested tuple/list/dictionary of `paddle.Tensor`: The gathered tensor(s). Note that the
            first dimension of the result is *num_processes* multiplied by the first dimension of the input tensors.

        Example:

        ```python
        >>> # Assuming four processes
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> process_tensor = torch.tensor([accelerator.process_index])
        >>> gathered_tensor = accelerator.gather(process_tensor)
        >>> gathered_tensor
        tensor([0, 1, 2, 3])
        ```
        """
        return gather(tensor)

    def gather_for_metrics(self, tensor):
        """
        Gathers `tensor` and potentially drops duplicates in the last batch if on a distributed system. Should be used
        for gathering the inputs and targets for metric calculation.

        Args:
            tensor (`paddle.Tensor`, or a nested tuple/list/dictionary of `paddle.Tensor`):
                The tensors for calculating metrics across all processes.

        Example:

        ```python
        >>> # Assuming two processes, with a batch size of 5 on a dataset with 9 samples
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> dataloader = paddle.io.DataLoader(range(9), batch_size=5)
        >>> dataloader = accelerator.prepare(dataloader)
        >>> batch = next(iter(dataloader))
        >>> gathered_items = accelerator.gather_for_metrics(batch)
        >>> len(gathered_items)
        9
        ```
        """
        tensor = self.gather(tensor)

        try:
            if self.gradient_state.end_of_dataloader:
                # at the end of a dataloader, `gather_for_metrics` regresses to
                # `gather` unless the dataset has a remainder so log.
                if self.gradient_state.remainder == -1:
                    logger.info(
                        "The used dataset had no length, returning gathered tensors. You should drop the remainder yourself."
                    )
                    return tensor
                elif self.gradient_state.remainder > 0:
                    # Last batch needs to be truncated on distributed systems as it contains additional samples
                    def _adjust_samples(tensor):
                        return tensor[: self.gradient_state.remainder]

                    return recursively_apply(_adjust_samples, tensor)
                else:  # remainder is 0
                    # no remainder even though at end of dataloader, so nothing to do.
                    return tensor
            else:
                # Not at the end of the dataloader, no need to adjust the tensors
                return tensor
        except Exception:
            # Dataset had no length or raised an error
            return tensor

    def reduce(self, tensor, reduction="sum"):
        """
        Reduce the values in *tensor* across all processes based on *reduction*.

        Note:
            All processes get the reduced value.

        Args:
            tensor (`paddle.Tensor`, or a nested tuple/list/dictionary of `paddle.Tensor`):
                The tensors to reduce across all processes.
            reduction (`str`, *optional*, defaults to "sum"):
                A reduction type, can be one of 'sum', 'mean', or 'none'. If 'none', will not perform any operation.

        Returns:
            `paddle.Tensor`, or a nested tuple/list/dictionary of `paddle.Tensor`:
                The reduced tensor(s).

        Example:

        ```python
        >>> # Assuming two processes
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> process_tensor = torch.arange(accelerator.num_processes) + 1 + (2 * accelerator.process_index)
        >>> process_tensor = process_tensor.to(accelerator.device)
        >>> reduced_tensor = accelerator.reduce(process_tensor, reduction="sum")
        >>> reduced_tensor
        tensor([4, 6])
        ```
        """
        return reduce(tensor, reduction)

    def pad_across_processes(self, tensor, dim=0, pad_index=0, pad_first=False):
        """
        Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
        they can safely be gathered.

        Args:
            tensor (nested list/tuple/dictionary of `paddle.Tensor`):
                The data to gather.
            dim (`int`, *optional*, defaults to 0):
                The dimension on which to pad.
            pad_index (`int`, *optional*, defaults to 0):
                The value with which to pad.
            pad_first (`bool`, *optional*, defaults to `False`):
                Whether to pad at the beginning or the end.

        Returns:
            `paddle.Tensor`, or a nested tuple/list/dictionary of `paddle.Tensor`:
                The padded tensor(s).

        Example:

        ```python
        >>> # Assuming two processes, with the first processes having a tensor of size 1 and the second of size 2
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> process_tensor = torch.arange(accelerator.process_index + 1).to(accelerator.device)
        >>> padded_tensor = accelerator.pad_across_processes(process_tensor)
        >>> padded_tensor.shape
        torch.Size([2])
        ```
        """
        return pad_across_processes(tensor, dim=dim, pad_index=pad_index, pad_first=pad_first)

    def unwrap_model(self, model, keep_fp32_wrapper: bool = True):
        """
        Unwraps the `model` from the additional layer possible added by [`~Accelerator.prepare`]. Useful before saving
        the model.

        Args:
            model (`paddle.nn.Layer`):
                The model to unwrap.
            keep_fp32_wrapper (`bool`, *optional*, defaults to `True`):
                Whether to not remove the mixed precision hook if it was added.

        Returns:
            `paddle.nn.Layer`: The unwrapped model.

        Example:

        ```python
        >>> # Assuming two GPU processes
        >>> from torch.nn.parallel import DistributedDataParallel
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> model = accelerator.prepare(MyModel())
        >>> print(model.__class__.__name__)
        DistributedDataParallel

        >>> model = accelerator.unwrap_model(model)
        >>> print(model.__class__.__name__)
        MyModel
        ```
        """
        return extract_model_from_parallel(model, keep_fp32_wrapper)

    def wait_for_everyone(self):
        """
        Will stop the execution of the current process until every other process has reached that point (so this does
        nothing when the script is only run in one process). Useful to do before saving a model.

        Example:

        ```python
        >>> # Assuming two GPU processes
        >>> import time
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> if accelerator.is_main_process:
        ...     time.sleep(2)
        >>> else:
        ...     print("I'm waiting for the main process to finish its sleep...")
        >>> accelerator.wait_for_everyone()
        >>> # Should print on every process at the same time
        >>> print("Everyone is here")
        ```
        """
        wait_for_everyone()

    @on_main_process
    def init_trackers(self, project_name: str, config: dict | None = None, init_kwargs: dict | None = {}):
        """
        Initializes a run for all trackers stored in `self.log_with`, potentially with starting configurations

        Args:
            project_name (`str`):
                The name of the project. All trackers will save their data based on this
            config (`dict`, *optional*):
                Optional starting configuration to be logged.
            init_kwargs (`dict`, *optional*):
                A nested dictionary of kwargs to be passed to a specific tracker's `__init__` function. Should be
                formatted like so:
                ```python
                {"wandb": {"tags": ["tag_a", "tag_b"]}}
                ```

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(log_with="tensorboard")
        >>> accelerator.init_trackers(
        ...     project_name="my_project",
        ...     config={"learning_rate": 0.001, "batch_size": 32},
        ...     init_kwargs={"tensorboard": {"flush_secs": 60}},
        ... )
        ```
        """
        self.trackers = []
        for tracker in self.log_with:
            if issubclass(type(tracker), GeneralTracker):
                # Custom trackers are already initialized
                self.trackers.append(tracker)
            else:
                tracker_init = LOGGER_TYPE_TO_CLASS[str(tracker)]
                if getattr(tracker_init, "requires_logging_directory"):
                    # We can skip this check since it was done in `__init__`
                    self.trackers.append(
                        tracker_init(project_name, self.logging_dir, **init_kwargs.get(str(tracker), {}))
                    )
                else:
                    self.trackers.append(tracker_init(project_name, **init_kwargs.get(str(tracker), {})))
        if config is not None:
            for tracker in self.trackers:
                tracker.store_init_configuration(config)

    def get_tracker(self, name: str, unwrap: bool = False):
        """
        Returns a `tracker` from `self.trackers` based on `name` on the main process only.

        Args:
            name (`str`):
                The name of a tracker, corresponding to the `.name` property.
            unwrap (`bool`):
                Whether to return the internal tracking mechanism or to return the wrapped tracker instead
                (recommended).

        Returns:
            `GeneralTracker`: The tracker corresponding to `name` if it exists.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(log_with="tensorboard")
        >>> accelerator.init_trackers("my_project")
        >>> tensorboard_tracker = accelerator.get_tracker("tensorboard")
        ```
        """
        if len(getattr(self, "trackers", [])) > 0:
            for tracker in self.trackers:
                if tracker.name == name:
                    return tracker.tracker if unwrap else tracker
            raise ValueError(f"{name} is not an available tracker stored inside the `Accelerator`.")
        # Handle tracker only made on main process
        return GeneralTracker(_blank=True)

    @on_main_process
    def log(self, values: dict, step: int | None = None, log_kwargs: dict | None = {}):
        """
        Logs `values` to all stored trackers in `self.trackers` on the main process only.

        Args:
            values (`dict`):
                Values should be a dictionary-like object containing only types `int`, `float`, or `str`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            log_kwargs (`dict`, *optional*):
                A nested dictionary of kwargs to be passed to a specific tracker's `log` function. Should be formatted
                like so:
                ```python
                {"wandb": {"tags": ["tag_a", "tag_b"]}}
                ```

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(log_with="tensorboard")
        >>> accelerator.init_trackers("my_project")
        >>> accelerator.log({"loss": 0.5, "accuracy": 0.9})
        ```
        """
        for tracker in self.trackers:
            tracker.log(values, step=step, **log_kwargs.get(tracker.name, {}))

    @on_main_process
    def end_training(self):
        """
        Runs any special end training behaviors, such as stopping trackers on the main process only. Should always be
        called at the end of your script if using experiment tracking.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(log_with="tensorboard")
        >>> accelerator.init_trackers("my_project")
        >>> # Do training
        >>> accelerator.end_training()
        ```
        """
        for tracker in self.trackers:
            tracker.finish()

    def save(self, obj, f):
        """
        Save the object passed to disk once per machine. Use in place of `torch.save`.

        Args:
            obj (`object`): The object to save.
            f (`str` or `os.PathLike`): Where to save the content of `obj`.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> arr = [0, 1, 2, 3]
        >>> accelerator.save(arr, "array.pkl")
        ```
        """
        save(obj, f)

    def save_model(
        self,
        model: paddle.nn.Layer,
        save_directory: Union[str, os.PathLike],
        max_shard_size: Union[int, str] = "10GB",
        safe_serialization: bool = False,
    ):
        """
        Save a model so that it can be re-loaded using load_checkpoint_in_model

        Arguments:
            model: (`paddle.nn.Layer`):
                Model to be saved. The model can be wrapped or unwraped.
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).

                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>

            safe_serialization (`bool`, *optional*, defaults to `False`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> model = ...
        >>> accelerator.save_model(model, save_directory)
        ```
        """

        if safe_serialization and not is_safetensors_available():
            raise ImportError("`safe_serialization` requires the `safetensors library: `pip install safetensors`.")

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # get the state_dict of the model
        state_dict = self.get_state_dict(model)

        if safe_serialization:
            # Safetensors does not allow tensor aliasing.
            # We're going to remove aliases before saving
            ptrs = collections.defaultdict(list)
            # when bnb serialization is used the weights in the state dict can be strings
            for name, tensor in state_dict.items():
                if not isinstance(tensor, str):
                    ptrs[id_tensor_storage(tensor)].append(name)

            # These are all the pointers of shared tensors.
            shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
            warn_names = set()
            for names in shared_ptrs.values():
                # When not all duplicates have been cleaned, still remove those keys, but put a clear warning.
                # If the link between tensors was done at runtime then `from_pretrained` will not get
                # the key back leading to random tensor. A proper warning will be shown
                # during reload (if applicable), but since the file is not necessarily compatible with
                # the config, better show a proper warning.
                found = 0
                for name in names:
                    if name in state_dict:
                        found += 1
                        if found > 1:
                            del state_dict[name]
                            warn_names.add(name)
            if len(warn_names) > 0:
                logger.warning_once(
                    f"Removed shared tensor {warn_names} while saving. This should be OK, but check by verifying that you don't receive any warning while reloading",
                )

        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME

        # Shard the model if it is too big.
        shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name=weights_name)

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "")

            # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
            filename_no_suffix = filename.replace(".bin", "")
            reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in shards.keys()
                and reg.fullmatch(filename_no_suffix) is not None
                and PartialState().is_main_process
            ):
                os.remove(full_filename)

        # Save the model
        for shard_file, shard in shards.items():
            self.save(shard, os.path.join(save_directory, shard_file))

        if index is None:
            path_to_weights = os.path.join(save_directory, WEIGHTS_NAME)
            logger.info(f"Model weights saved in {path_to_weights}")
        else:
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, save_index_file)
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )

    def register_save_state_pre_hook(self, hook: Callable[..., None]) -> hooks.RemovableHandle:
        """
        Registers a pre hook to be run before `save_checkpoint` is called in [`Accelerator.save_state`].

        Args:
            hook (`Callable`):
                A function to be called in [`Accelerator.save_state`] before `save_checkpoint`.

        The hook should have the following signature:

        `hook(models: list[paddle.nn.Layer], weights: list[dict[str, paddle.Tensor]], input_dir: str) -> None`

        The `models` argument are the models as saved in the accelerator state under `accelerator._models`, `weigths`
        argument are the state dicts of the `models`, and the `input_dir` argument is the `input_dir` argument passed
        to [`Accelerator.load_state`].

        <Tip>

        Should only be used in conjunction with [`Accelerator.register_load_state_pre_hook`]. Can be useful to save
        configurations in addition to model weights. Can also be used to overwrite model saving with a customized
        method. In this case, make sure to remove already loaded weights from the weights list.

        </Tip>

        Returns:
            `torch.utils.hooks.RemovableHandle`: a handle that can be used to remove the added hook by calling
            `handle.remove()`
        """
        handle = hooks.RemovableHandle(self._save_model_state_pre_hook)
        self._save_model_state_pre_hook[handle.id] = hook
        return handle

    def save_state(self, output_dir: str = None, **save_model_func_kwargs):
        """
        Saves the current states of the model, optimizer, scaler, RNG generators, and registered objects to a folder.

        If a `ProjectConfiguration` was passed to the `Accelerator` object with `automatic_checkpoint_naming` enabled
        then checkpoints will be saved to `self.project_dir/checkpoints`. If the number of current saves is greater
        than `total_limit` then the oldest save is deleted. Each checkpoint is saved in seperate folders named
        `checkpoint_<iteration>`.

        Otherwise they are just saved to `output_dir`.

        <Tip>

        Should only be used when wanting to save a checkpoint during training and restoring the state in the same
        environment.

        </Tip>

        Args:
            output_dir (`str` or `os.PathLike`):
                The name of the folder to save all relevant weights and states.
            save_model_func_kwargs (`dict`, *optional*):
                Additional keyword arguments for saving model which can be passed to the underlying save function, such
                as optional arguments for DeepSpeed's `save_checkpoint` function.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> model, optimizer, lr_scheduler = ...
        >>> model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
        >>> accelerator.save_state(output_dir="my_checkpoint")
        ```
        """
        if self.project_configuration.automatic_checkpoint_naming:
            output_dir = os.path.join(self.project_dir, "checkpoints")
        os.makedirs(output_dir, exist_ok=True)
        if self.project_configuration.automatic_checkpoint_naming:
            folders = [os.path.join(output_dir, folder) for folder in os.listdir(output_dir)]
            if self.project_configuration.total_limit is not None and (
                len(folders) + 1 > self.project_configuration.total_limit
            ):

                def _inner(folder):
                    return list(map(int, re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)))[0]

                folders.sort(key=_inner)
                logger.warning(
                    f"Deleting {len(folders) + 1 - self.project_configuration.total_limit} checkpoints to make room for new checkpoint."
                )
                for folder in folders[: len(folders) + 1 - self.project_configuration.total_limit]:
                    shutil.rmtree(folder)
            output_dir = os.path.join(output_dir, f"checkpoint_{self.save_iteration}")
            if os.path.exists(output_dir):
                raise ValueError(
                    f"Checkpoint directory {output_dir} ({self.save_iteration}) already exists. Please manually override `self.save_iteration` with what iteration to start with."
                )
            self.wait_for_everyone()
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving current state to {output_dir}")

        # Save the models taking care of FSDP and DeepSpeed nuances
        weights = []
        for i, model in enumerate(self._models):
            weights.append(self.get_state_dict(model, unwrap=False))

        # Save the optimizers taking care of FSDP and DeepSpeed nuances
        optimizers = self._optimizers

        # Save the lr schedulers taking care of DeepSpeed nuances
        schedulers = []
        schedulers = self._schedulers

        # Call model loading hooks that might have been registered with
        # accelerator.register_model_state_hook
        for hook in self._save_model_state_pre_hook.values():
            hook(self._models, weights, output_dir)

        save_location = save_accelerator_state(
            output_dir, weights, optimizers, schedulers, self.state.process_index, self.scaler
        )
        for i, obj in enumerate(self._custom_objects):
            save_custom_state(obj, output_dir, i)
        self.project_configuration.iteration += 1
        return save_location

    def register_load_state_pre_hook(self, hook: Callable[..., None]) -> hooks.RemovableHandle:
        """
        Registers a pre hook to be run before [`load_checkpoint`] is called in [`Accelerator.load_state`].

        Args:
            hook (`Callable`):
                A function to be called in [`Accelerator.load_state`] before `load_checkpoint`.

        The hook should have the following signature:

        `hook(models: list[paddle.nn.Layer], input_dir: str) -> None`

        The `models` argument are the models as saved in the accelerator state under `accelerator._models`, and the
        `input_dir` argument is the `input_dir` argument passed to [`Accelerator.load_state`].

        <Tip>

        Should only be used in conjunction with [`Accelerator.register_save_state_pre_hook`]. Can be useful to load
        configurations in addition to model weights. Can also be used to overwrite model loading with a customized
        method. In this case, make sure to remove already loaded models from the models list.

        </Tip>

        Returns:
            `torch.utils.hooks.RemovableHandle`: a handle that can be used to remove the added hook by calling
            `handle.remove()`
        """
        handle = hooks.RemovableHandle(self._load_model_state_pre_hook)
        self._load_model_state_pre_hook[handle.id] = hook
        return handle

    def load_state(self, input_dir: str = None, **load_model_func_kwargs):
        """
        Loads the current states of the model, optimizer, scaler, RNG generators, and registered objects.

        <Tip>

        Should only be used in conjunction with [`Accelerator.save_state`]. If a file is not registered for
        checkpointing, it will not be loaded if stored in the directory.

        </Tip>

        Args:
            input_dir (`str` or `os.PathLike`):
                The name of the folder all relevant weights and states were saved in. Can be `None` if
                `automatic_checkpoint_naming` is used, and will pick up from the latest checkpoint.
            load_model_func_kwargs (`dict`, *optional*):
                Additional keyword arguments for loading model which can be passed to the underlying load function,
                such as optional arguments for DeepSpeed's `load_checkpoint` function or a `map_location` to load the
                model and optimizer on.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> model, optimizer, lr_scheduler = ...
        >>> model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
        >>> accelerator.load_state("my_checkpoint")
        ```
        """
        if input_dir is not None:
            # Check if folder exists
            input_dir = os.path.expanduser(input_dir)
            if not os.path.isdir(input_dir):
                raise ValueError(f"Tried to find {input_dir} but folder does not exist")
        elif self.project_configuration.automatic_checkpoint_naming:
            # Pick up from automatic checkpoint naming
            input_dir = os.path.join(self.project_dir, "checkpoints")
            folders = [os.path.join(input_dir, folder) for folder in os.listdir(input_dir)]

            def _inner(folder):
                return list(map(int, re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)))[0]

            folders.sort(key=_inner)
            input_dir = os.path.join(input_dir, folders[-1])
        else:
            raise ValueError("No input_dir provided and automatic checkpoint naming is disabled.")
        logger.info(f"Loading states from {input_dir}")

        # Load the models taking care of FSDP and DeepSpeed nuances
        models = []
        for i, model in enumerate(self._models):
            models.append(model)

        # Load the optimizers taking care of FSDP and DeepSpeed nuances
        optimizers = self._optimizers

        # Load the lr schedulers taking care of DeepSpeed nuances
        schedulers = self._schedulers

        # Call model loading hooks that might have been registered with
        # accelerator.register_model_state_hook
        for hook in self._load_model_state_pre_hook.values():
            hook(models, input_dir)

        map_location = load_model_func_kwargs.pop("map_location", None)
        if map_location is None:
            if self.num_processes > 1 and self.distributed_type in (
                DistributedType.MULTI_GPU,
                DistributedType.MULTI_NPU,
            ):
                map_location = "on_device"
            else:
                map_location = "cpu"

        load_accelerator_state(
            input_dir,
            models,
            optimizers,
            schedulers,
            self.state.process_index,
            self.scaler,
            map_location,
            **load_model_func_kwargs,
        )
        custom_checkpoints = [
            f for f in os.listdir(input_dir) if re.search(r"^custom_checkpoint_\d+\.pkl$", f) is not None
        ]
        if len(custom_checkpoints) != len(self._custom_objects):
            err = "Number of custom checkpoints in folder {input_dir} does not match the number of registered objects:"
            err += f"\n\tFound checkpoints: {len(custom_checkpoints)}"
            err += f"\n\tRegistered objects: {len(self._custom_objects)}\n"
            err += "Please make sure to only load checkpoints from folders that were created with the same set of registered objects,"
            err += "or avoid using `custom_checkpoint` in the filename for files in that same directory and load them in manually."
            raise RuntimeError(err)
        else:
            logger.info(f"Loading in {len(custom_checkpoints)} custom states")
            for index, obj in enumerate(self._custom_objects):
                load_custom_state(obj, input_dir, index)

    def free_memory(self):
        """
        Will release all references to the internal objects stored and call the garbage collector. You should call this
        method between two trainings with different models/optimizers. Also will reset `Accelerator.step` to 0.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> model, optimizer, scheduler = ...
        >>> model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        >>> accelerator.free_memory()
        >>> del model, optimizer, scheduler
        ```
        """
        self._schedulers = []
        self._optimizers = []
        self._models = []
        self._dataloaders = []
        self.deepspeed_engine_wrapped = None
        self.step = 0
        release_memory()

    def clear(self):
        """
        Alias for [`Accelerate.free_memory`], releases all references to the internal objects stored and call the
        garbage collector. You should call this method between two trainings with different models/optimizers.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> model, optimizer, scheduler = ...
        >>> model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        >>> accelerator.free_memory()
        >>> del model, optimizer, scheduler
        ```
        """
        self.free_memory()

    def _get_named_parameters(self, *args):
        named_parameters = {}
        for obj in args:
            if isinstance(obj, paddle.nn.Layer):
                obj = extract_model_from_parallel(obj)
                named_parameters.update({n: p for n, p in obj.named_parameters()})
        return named_parameters

    def _get_devices(self, *args):
        model_device = None
        optimizer_device = None
        for obj in args:
            # Loop through model parameters and stop at the first once we have its device.
            if isinstance(obj, paddle.nn.Layer):
                for param in obj.parameters():
                    model_device = param.place
                    break
            # Loop through optimizer parameters groups and stop at the first once we have its device.
            if isinstance(obj, paddle.optimizer.Optimizer):
                for param in obj._parameter_list:
                    optimizer_device = param.place
                    break
        return (model_device, optimizer_device)

    def get_state_dict(self, model, unwrap=True):
        """
        Returns the state dictionary of a model sent through [`Accelerator.prepare`] potentially without full
        precision.

        Args:
            model (`paddle.nn.Layer`):
                A PyTorch model sent through [`Accelerator.prepare`]
            unwrap (`bool`, *optional*, defaults to `True`):
                Whether to return the original underlying state_dict of `model` or to return the wrapped state_dict

        Returns:
            `dict`: The state dictionary of the model potentially without full precision.

        Example:

        ```python
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> net = torch.nn.Linear(2, 2)
        >>> net = accelerator.prepare(net)
        >>> state_dict = accelerator.get_state_dict(net)
        ```
        """


        if unwrap:
            model = self.unwrap_model(model)
        state_dict = model.state_dict()

        if state_dict is not None:
            for k in state_dict:
                if getattr(state_dict[k], "dtype", None) == paddle.float16:
                    state_dict[k] = state_dict[k]._to(dtype="float32")

        return state_dict

    def register_for_checkpointing(self, *objects):
        """
        Makes note of `objects` and will save or load them in during `save_state` or `load_state`.

        These should be utilized when the state is being loaded or saved in the same script. It is not designed to be
        used in different scripts.

        <Tip>

        Every `object` must have a `load_state_dict` and `state_dict` function to be stored.

        </Tip>

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> # Assume `CustomObject` has a `state_dict` and `load_state_dict` function.
        >>> obj = CustomObject()
        >>> accelerator.register_for_checkpointing(obj)
        >>> accelerator.save_state("checkpoint.pt")
        ```
        """
        invalid_objects = []
        for obj in objects:
            if not hasattr(obj, "state_dict") or not hasattr(obj, "load_state_dict"):
                invalid_objects.append(obj)
        if len(invalid_objects) > 0:
            err = "All `objects` must include a `state_dict` and `load_state_dict` function to be stored. The following inputs are invalid:"
            for index, obj in enumerate(invalid_objects):
                err += f"\n\t- Item at index {index}, `{get_pretty_name(obj)}`"
            raise ValueError(err)
        self._custom_objects.extend(objects)

    @contextmanager
    def autocast(self, cache_enabled: bool = False, autocast_handler: AutocastKwargs = None):
        """
        Will apply automatic mixed-precision inside the block inside this context manager, if it is enabled. Nothing
        different will happen otherwise.

        A different `autocast_handler` can be passed in to override the one set in the `Accelerator` object. This is
        useful in blocks under `autocast` where you want to revert to fp32.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(mixed_precision="fp16")
        >>> with accelerator.autocast():
        ...     train()
        ```
        """
        if cache_enabled:
            warnings.warn(
                "Passing `cache_enabled=True` to `accelerator.autocast` is deprecated and will be removed in v0.23.0. "
                "Please use the `AutocastKwargs` class instead and pass it to the `Accelerator` as a `kwarg_handler`.",
                FutureWarning,
            )
            if self.autocast_handler is not None:
                self.autocast_handler.cache_enabled = True
            else:
                self.autocast_handler = AutocastKwargs()
        if autocast_handler is None:
            autocast_handler = self.autocast_handler
        autocast_context = get_mixed_precision_context_manager(self.native_amp, autocast_handler)
        autocast_context.__enter__()
        yield
        autocast_context.__exit__(*sys.exc_info())

    @property
    def optimizer_step_was_skipped(self):
        """
        Whether or not the optimizer update was skipped (because of gradient overflow in mixed precision), in which
        case the learning rate should not be changed.
        """
        for optimizer in self._optimizers:
            if optimizer.step_was_skipped:
                return True
        return False

    def skip_first_batches(self, dataloader, num_batches: int = 0):
        """
        Creates a new `paddle.io.DataLoader` that will efficiently skip the first `num_batches`.

        Args:
            dataloader (`paddle.io.DataLoader`): The data loader in which to skip batches.
            num_batches (`int`, *optional*, defaults to 0): The number of batches to skip

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)
        >>> skipped_dataloader = accelerator.skip_first_batches(dataloader, num_batches=2)
        >>> # for the first epoch only
        >>> for input, target in skipped_dataloader:
        ...     optimizer.zero_grad()
        ...     output = model(input)
        ...     loss = loss_func(output, target)
        ...     accelerator.backward(loss)
        ...     optimizer.step()

        >>> # subsequent epochs
        >>> for input, target in dataloader:
        ...     optimizer.zero_grad()
        ...     ...
        ```
        """
        return dataloader

    def __deepcopy__(self, memo):
        logger.info("Deep copying the `Accelerator` object, note that this will point to the same original object.")
        return self

    def verify_device_map(self, model: paddle.nn.Layer) -> bool:
        """
        Verifies that `model` has not been prepared with big model inference with a device-map resembling `auto`.
        """
        # Checks if any of the child module has the attribute `hf_device_map`.
        has_hf_device_map = any(hasattr(m, "hf_device_map") for m in model.sublayers(include_self=True))

        return has_hf_device_map
