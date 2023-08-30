import paddle
import numpy as np

old_copy_ = paddle.Tensor.copy_
def copy_(self, x, non_blocking=False):
    return old_copy_(self, x, non_blocking)

paddle.Tensor.copy_ = copy_

def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)

def masked_fill_(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return x.copy_(paddle.where(mask, y, x), False)

if not hasattr(paddle.Tensor, "masked_fill"):
    paddle.Tensor.masked_fill = masked_fill
if not hasattr(paddle.Tensor, "masked_fill_"):
    paddle.Tensor.masked_fill_ = masked_fill_
    
if not hasattr(paddle, "masked_fill"):
    paddle.masked_fill = masked_fill

if not hasattr(paddle, "clamp"):
    paddle.clamp = paddle.clip
if not hasattr(paddle.Tensor, "clamp"):
    paddle.Tensor.clamp = paddle.Tensor.clip

def finfo(dtype: paddle.dtype=None):
    if dtype is None:
        dtype = paddle.get_default_dtype()

    if dtype in [paddle.bfloat16, "bfloat16"]:
        # Numpy do not support `np.finfo(np.uint16)`, so try to construct a finfo object to fetch min value
        class BFloatFInfo:
            min = -3.3895313892515355e38

        return BFloatFInfo
    if dtype in [paddle.float32, "float32"]:
        return np.finfo(np.float32)
    if dtype in [paddle.float16, "float16"]:
        return np.finfo(np.float16)
    if dtype in [paddle.float64, "float64"]:
        return np.finfo(np.float64)

if not hasattr(paddle, "finfo"):
    paddle.finfo = finfo
    
if not hasattr(paddle.Tensor, "triu"):
    paddle.Tensor.triu = paddle.triu

def tensor_to(self, dtype=None, device=None, blocking=None):
    if isinstance(dtype, paddle.dtype):
        pass
    elif isinstance(dtype, str):
        if "pu" in str(dtype):
            device = dtype
            dtype = None

    if device is not None and isinstance(device, core.Place):
        device = re.findall(r'Place\((.*)\)', str(device))[0]
    return self._to(dtype=dtype, device=device, blocking=blocking)

if not hasattr(paddle.Tensor, "to"):
    paddle.Tensor.to = tensor_to

def mul_(self, x):
    self.copy_(self * x, False)
    return self

if not hasattr(paddle.Tensor, "mul_"):
    paddle.Tensor.mul_ = mul_

def div_(self, x):
    self.copy_(self / x, False)
    return self

if not hasattr(paddle.Tensor, "div_"):
    paddle.Tensor.div_ = div_

def split_new(x, size, axis=-1):
    sb = [size] * (x.shape[axis] // size)
    return paddle.split(x, sb, axis=axis)

if not hasattr(paddle.Tensor, "split_new"):
    paddle.Tensor.split_new = split_new

def requires_grad_(self, value=True):
    for v in self.parameters():
        v.stop_gradient = not value

if not hasattr(paddle.nn.Layer, "requires_grad_"):
    paddle.nn.Layer.requires_grad_ = requires_grad_
    
import paddle
import paddle.nn as nn
import itertools
from typing import List, Any, Mapping, Callable, Optional, Dict
from collections import OrderedDict, namedtuple
import functools
import weakref


class RemovableHandle:
    r"""
    A handle which provides the capability to remove a hook.

    Args:
        hooks_dict (dict): A dictionary of hooks, indexed by hook ``id``.
        extra_dict (dict): An additional dictionary whose keys will be deleted
            when the same keys are removed from ``hooks_dict``.
    """

    id: int
    next_id: int = 0

    def __init__(self, hooks_dict: Any, *, extra_dict: Any = None) -> None:
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

        self.extra_dict_ref = (
            weakref.ref(extra_dict)
            if extra_dict is not None
            else None
        )

    def remove(self) -> None:
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

        if self.extra_dict_ref is not None:
            extra_dict = self.extra_dict_ref()
            if extra_dict is not None and self.id in extra_dict:
                del extra_dict[self.id]

    def __getstate__(self):
        return (
            (self.hooks_dict_ref(), self.id)
            if self.extra_dict_ref is None
            else (self.hooks_dict_ref(), self.id, self.extra_dict_ref())
        )

    def __setstate__(self, state) -> None:
        if state[0] is None:
            # create a dead reference
            self.hooks_dict_ref = weakref.ref(OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

        self.extra_dict_ref = (
            None
            if len(state) < 3
            else weakref.ref(OrderedDict() if state[2] is None else state[2])
        )

    def __enter__(self) -> "RemovableHandle":
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        self.remove()


_EXTRA_STATE_KEY_SUFFIX = '_extra_state'

nn.Layer.raw__init__ = nn.Layer.__init__


def new_init(self, *args, **kwargs) -> Any:
    self.__load_state_dict_pre_hooks = OrderedDict()
    self.__load_state_dict_post_hooks = OrderedDict()
    self.__state_dict_pre_hooks = OrderedDict()
    self.__state_dict_hooks = OrderedDict()
    nn.Layer.raw__init__(self, *args, **kwargs)


nn.Layer.__init__ = new_init
nn.Layer._version = 1


class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super().__repr__()

    __str__ = __repr__


def get_extra_state(self) -> Any:
    """
    Returns any extra state to include in the module's state_dict.
    Implement this and a corresponding :func:`set_extra_state` for your module
    if you need to store extra state. This function is called when building the
    module's `state_dict()`.

    Note that extra state should be picklable to ensure working serialization
    of the state_dict. We only provide provide backwards compatibility guarantees
    for serializing Tensors; other objects may break backwards compatibility if
    their serialized pickled form changes.

    Returns:
        object: Any extra state to store in the module's state_dict
    """
    raise RuntimeError(
        "Reached a code path in Module.get_extra_state() that should never be called. "
        "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
        "to report this bug.")


def set_extra_state(self, state: Any):
    """
    This function is called from :func:`load_state_dict` to handle any extra state
    found within the `state_dict`. Implement this function and a corresponding
    :func:`get_extra_state` for your module if you need to store extra state within its
    `state_dict`.

    Args:
        state (dict): Extra state from the `state_dict`
    """
    raise RuntimeError(
        "Reached a code path in Module.set_extra_state() that should never be called. "
        "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
        "to report this bug.")


def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                          missing_keys, unexpected_keys, error_msgs):
    r"""Copies parameters and buffers from :attr:`state_dict` into only
    this module, but not its descendants. This is called on every submodule
    in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
    module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
    For state dicts without metadata, :attr:`local_metadata` is empty.
    Subclasses can achieve class-specific backward compatible loading using
    the version number at `local_metadata.get("version", None)`.

    .. note::
        :attr:`state_dict` is not the same object as the input
        :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
        it can be modified.

    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        prefix (str): the prefix for parameters and buffers used in this
            module
        local_metadata (dict): a dict containing the metadata for this module.
            See
        strict (bool): whether to strictly enforce that the keys in
            :attr:`state_dict` with :attr:`prefix` match the names of
            parameters and buffers in this module
        missing_keys (list of str): if ``strict=True``, add missing keys to
            this list
        unexpected_keys (list of str): if ``strict=True``, add unexpected
            keys to this list
        error_msgs (list of str): error messages should be added to this
            list, and will be reported together in
            :meth:`~torch.nn.Module.load_state_dict`
    """
    for hook in self.__load_state_dict_pre_hooks.values():
        hook(state_dict, prefix, local_metadata, strict,
             missing_keys, unexpected_keys, error_msgs)

    persistent_buffers = {k: v for k, v in self._buffers.items(
    ) if k not in self._non_persistable_buffer_names_set}
    local_name_params = itertools.chain(
        self._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}

    for name, param in local_state.items():
        key = prefix + name
        if key in state_dict:
            input_param = state_dict[key]
            if not paddle.is_tensor(input_param):
                input_param = paddle.to_tensor(
                    input_param, dtype=input_param.dtype)
            is_param_lazy = False
            if not is_param_lazy and input_param.shape != param.shape:
                # local shape should match the one in checkpoint
                error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                  'the shape in current model is {}.'
                                  .format(key, input_param.shape, param.shape))
                continue
            try:
                with paddle.no_grad():
                    param.copy_(input_param, False)
            except Exception as ex:
                error_msgs.append('While copying the parameter named "{}", '
                                  'whose dimensions in the model are {} and '
                                  'whose dimensions in the checkpoint are {}, '
                                  'an exception occurred : {}.'
                                  .format(key, param.shape, input_param.shape, ex.args))
        elif strict:
            missing_keys.append(key)

    extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
    if getattr(self.__class__, "set_extra_state", nn.Layer.set_extra_state) is not nn.Layer.set_extra_state:
        if extra_state_key in state_dict:
            self.set_extra_state(state_dict[extra_state_key])
        elif strict:
            missing_keys.append(extra_state_key)
    elif strict and (extra_state_key in state_dict):
        unexpected_keys.append(extra_state_key)

    if strict:
        for key in state_dict.keys():
            if key.startswith(prefix) and key != extra_state_key:
                input_name = key[len(prefix):]
                # get the name of param/buffer/child
                input_name = input_name.split('.', 1)[0]
                if input_name not in self._sub_layers and input_name not in local_state:
                    unexpected_keys.append(key)


def load_state_dict(self, state_dict: Mapping[str, Any],
                    strict: bool = True):
    r"""Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True``, then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :meth:`~torch.nn.Module.state_dict` function.

    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        strict (bool, optional): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys

    Note:
        If a parameter or buffer is registered as ``None`` and its corresponding key
        exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
        ``RuntimeError``.
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "Expected state_dict to be dict-like, got {}.".format(type(state_dict)))

    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    error_msgs: List[str] = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = OrderedDict(state_dict)
    if metadata is not None:
        # mypy isn't aware that "_metadata" exists in state_dict
        state_dict._metadata = metadata  # type: ignore[attr-defined]

    def load(module, local_state_dict, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            local_state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._sub_layers.items():
            if child is not None:
                child_prefix = prefix + name + '.'
                child_state_dict = {
                    k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                load(child, child_state_dict, child_prefix)

        # Note that the hook can modify missing_keys and unexpected_keys.
        incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
        for hook in module.__load_state_dict_post_hooks.values():
            out = hook(module, incompatible_keys)
            assert out is None, (
                "Hooks registered with ``register_load_state_dict_post_hook`` are not"
                "expected to return new values, if incompatible_keys need to be modified,"
                "it should be done inplace."
            )

    load(self, state_dict)
    del load

    if strict:
        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0, 'Unexpected key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(
                0, 'Missing key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in missing_keys)))

    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
            self.__class__.__name__, "\n\t".join(error_msgs)))
    return _IncompatibleKeys(missing_keys, unexpected_keys)


def _state_dict(self, *args, destination=None, prefix='', keep_vars=False):
    r"""Returns a dictionary containing references to the whole state of the module.

    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.
    Parameters and buffers set to ``None`` are not included.

    .. note::
        The returned object is a shallow copy. It contains references
        to the module's parameters and buffers.

    .. warning::
        Currently ``state_dict()`` also accepts positional arguments for
        ``destination``, ``prefix`` and ``keep_vars`` in order. However,
        this is being deprecated and keyword arguments will be enforced in
        future releases.

    .. warning::
        Please avoid the use of argument ``destination`` as it is not
        designed for end-users.

    Args:
        destination (dict, optional): If provided, the state of module will
            be updated into the dict and the same object is returned.
            Otherwise, an ``OrderedDict`` will be created and returned.
            Default: ``None``.
        prefix (str, optional): a prefix added to parameter and buffer
            names to compose the keys in state_dict. Default: ``''``.
        keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
            returned in the state dict are detached from autograd. If it's
            set to ``True``, detaching will not be performed.
            Default: ``False``.

    Returns:
        dict:
            a dictionary containing a whole state of the module

    Example::

        >>> # xdoctest: +SKIP("undefined vars")
        >>> module.state_dict().keys()
        ['bias', 'weight']

    """

    # TODO: Remove `args` and the parsing logic when BC allows.
    if len(args) > 0:
        if destination is None:
            destination = args[0]
        if len(args) > 1 and prefix == '':
            prefix = args[1]
        if len(args) > 2 and keep_vars is False:
            keep_vars = args[2]
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()

    local_metadata = dict(version=self._version)
    if hasattr(destination, "_metadata"):
        destination._metadata[prefix[:-1]] = local_metadata

    self._save_to_state_dict(destination, prefix, keep_vars)
    for name, module in self._sub_layers.items():
        if module is not None:
            module._state_dict(destination=destination,
                               prefix=prefix + name + '.', keep_vars=keep_vars)
    for hook in self.__state_dict_hooks.values():
        hook_result = hook(self, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination


def _save_to_state_dict(self, destination, prefix, keep_vars):
    r"""Saves module state to `destination` dictionary, containing a state
    of the module, but not its descendants. This is called on every
    submodule in :meth:`~torch.nn.Module.state_dict`.

    In rare cases, subclasses can achieve class-specific behavior by
    overriding this method with custom logic.

    Args:
        destination (dict): a dict where state will be stored
        prefix (str): the prefix for parameters and buffers used in this
            module
    """
    for hook in self.__state_dict_pre_hooks.values():
        hook(self, prefix, keep_vars)

    for name, param in self._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in self._buffers.items():
        if buf is not None and name not in self._non_persistable_buffer_names_set:
            destination[prefix + name] = buf if keep_vars else buf.detach()
    extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
    if getattr(self.__class__, "get_extra_state", nn.Layer.get_extra_state) is not nn.Layer.get_extra_state:
        destination[extra_state_key] = self.get_extra_state()


def get_sublayer(self, target: str):
    if target == "":
        return self

    atoms: List[str] = target.split(".")
    mod: nn.Layer = self

    for item in atoms:
        if not hasattr(mod, item):
            raise AttributeError(mod.__class__.__name__ +
                                 " has no " "attribute `" + item + "`")

        mod = getattr(mod, item)

        if not isinstance(mod, nn.Layer):
            raise AttributeError("`" + item + "` is not " "an nn.Layer")
    return mod


class _WrappedHook:
    def __init__(self, hook: Callable, module: Optional["nn.Layer"] = None):
        self.hook: Callable = hook
        functools.update_wrapper(self, hook)

        self.with_module: bool = False

        if module is not None:
            self.module: weakref.ReferenceType["nn.Layer"] = weakref.ref(
                module)
            self.with_module = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.with_module:
            module = self.module()
            if module is None:
                raise RuntimeError(
                    "You are trying to call the hook of a dead Module!")
            return self.hook(module, *args, **kwargs)
        return self.hook(*args, **kwargs)

    def __getstate__(self) -> Dict:
        result = {"hook": self.hook, "with_module": self.with_module}
        if self.with_module:
            result["module"] = self.module()

        return result

    def __setstate__(self, state: Dict):
        self.hook = state["hook"]
        self.with_module = state["with_module"]

        if self.with_module:
            if state["module"] is None:
                raise RuntimeError(
                    "You are trying to revive the hook of a dead Module!")
            self.module = weakref.ref(state["module"])


def register_load_state_dict_post_hook(self, hook):
    r"""Registers a post hook to be run after module's ``load_state_dict``
    is called.

    It should have the following signature::
        hook(module, incompatible_keys) -> None

    The ``module`` argument is the current module that this hook is registered
    on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
    of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
    is a ``list`` of ``str`` containing the missing keys and
    ``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

    The given incompatible_keys can be modified inplace if needed.

    Note that the checks performed when calling :func:`load_state_dict` with
    ``strict=True`` are affected by modifications the hook makes to
    ``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
    set of keys will result in an error being thrown when ``strict=True``, and
    clearing out both missing and unexpected keys will avoid an error.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(self.__load_state_dict_post_hooks)
    self.__load_state_dict_post_hooks[handle.id] = hook
    return handle


def _register_load_state_dict_pre_hook(self, hook, with_module=False):
    r"""These hooks will be called with arguments: `state_dict`, `prefix`,
    `local_metadata`, `strict`, `missing_keys`, `unexpected_keys`,
    `error_msgs`, before loading `state_dict` into `self`. These arguments
    are exactly the same as those of `_load_from_state_dict`.

    If ``with_module`` is ``True``, then the first argument to the hook is
    an instance of the module.

    Arguments:
        hook (Callable): Callable hook that will be invoked before
            loading the state dict.
        with_module (bool, optional): Whether or not to pass the module
            instance to the hook as the first parameter.
    """
    handle = RemovableHandle(self.__load_state_dict_pre_hooks)
    self.__load_state_dict_pre_hooks[handle.id] = _WrappedHook(
        hook, self if with_module else None)
    return handle


def register_state_dict_pre_hook(self, hook):
    r"""These hooks will be called with arguments: ``self``, ``prefix``,
    and ``keep_vars`` before calling ``state_dict`` on ``self``. The registered
    hooks can be used to perform pre-processing before the ``state_dict``
    call is made.
    """
    handle = RemovableHandle(self.__state_dict_pre_hooks)
    self.__state_dict_pre_hooks[handle.id] = hook
    return handle


def _register_state_dict_hook(self, hook):
    r"""These hooks will be called with arguments: `self`, `state_dict`,
    `prefix`, `local_metadata`, after the `state_dict` of `self` is set.
    Note that only parameters and buffers of `self` or its children are
    guaranteed to exist in `state_dict`. The hooks may modify `state_dict`
    inplace or return a new one.
    """
    handle = RemovableHandle(self.__state_dict_hooks)
    self.__state_dict_hooks[handle.id] = hook
    return handle


nn.Layer.register_state_dict_pre_hook = register_state_dict_pre_hook
nn.Layer.register_load_state_dict_post_hook = register_load_state_dict_post_hook
nn.Layer.get_sublayer = get_sublayer
nn.Layer.get_extra_state = get_extra_state
nn.Layer.set_extra_state = set_extra_state
nn.Layer.load_state_dict = load_state_dict
nn.Layer._save_to_state_dict = _save_to_state_dict
nn.Layer._state_dict = _state_dict
nn.Layer._load_from_state_dict = _load_from_state_dict
nn.Layer._register_state_dict_hook = _register_state_dict_hook
nn.Layer._register_load_state_dict_pre_hook = _register_load_state_dict_pre_hook


paddle.Tensor.contiguous = lambda x: x

from paddle.framework import core
import re

def layer_to(self, device=None, dtype=None, blocking=None, floating_only=True):
    if device is not None and isinstance(device, core.Place):
        device = re.findall(r'Place\((.*)\)', str(device))[0]
    return self._to_impl(
            device=device,
            dtype=dtype,
            blocking=blocking,
            include_sublayers=True,
            floating_only=floating_only,
        )
nn.Layer.to = layer_to