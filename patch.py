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

