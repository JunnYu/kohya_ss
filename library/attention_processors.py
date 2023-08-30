import math
from typing import Any
from einops import rearrange
import paddle
from ppdiffusers.models.attention_processor import Attention
import numpy as np

old_copy_ = paddle.Tensor.copy_
def copy_(self, x, non_blocking=False):
    return old_copy_(self, x, non_blocking)

paddle.Tensor.copy_ = copy_

if not hasattr(paddle.Tensor, "float"):
    paddle.Tensor.float = lambda x: x._to(dtype="float32")

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

    if device is not None and "Place" in str(device):
        device = str(device).lstrip("Place(").rstrip(")")
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

########################################################
import math
import paddle
from paddle import nn, einsum
from paddle.autograd import PyLayer

from einops import rearrange

# constants

EPSILON = 1e-10

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# flash attention forwards and backwards

# flash attention v1 - https://arxiv.org/abs/2205.14135
# flash attention v2 - https://tridao.me/publications/flash2/flash2.pdf

class FlashAttentionFunction(PyLayer):
    @staticmethod
    @paddle.no_grad()
    def forward(ctx, q, k, v, mask, causal, q_bucket_size, k_bucket_size):
        """ Algorithm 1 in the v2 paper """

        max_neg_value = -paddle.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        o = paddle.zeros_like(q)
        all_row_sums = paddle.zeros((*q.shape[:-1], 1),)
        all_row_maxes = paddle.full((*q.shape[:-1], 1), max_neg_value,)

        scale = (q.shape[-1] ** -0.5)

        num_row_tiles = math.ceil(q.shape[-2] / q_bucket_size)
        num_col_tiles = math.ceil(k.shape[-2] / k_bucket_size)

        if exists(mask) and mask.ndim == 2:
            mask = rearrange(mask, 'b n -> b 1 1 n')

        if not exists(mask):
            col_masks = (None,) * num_col_tiles
            mask = (col_masks,) * num_row_tiles 
        else:
            mask = ((mask,) * num_row_tiles) if mask.shape[-2] == 1 else mask.split_new(q_bucket_size, axis = -2)
            mask = tuple(((row_mask,) * num_col_tiles) if row_mask.shape[-1] == 1 else row_mask.split_new(k_bucket_size, axis = -1) for row_mask in mask)

        row_splits = zip(
            q.split_new(q_bucket_size, axis = -2),
            o.split_new(q_bucket_size, axis = -2), # 1
            mask,
            all_row_sums.split_new(q_bucket_size, axis = -2), # 1
            all_row_maxes.split_new(q_bucket_size, axis = -2), # 1
        )
        all_all_oc =[]
        all_all_row_sums = []
        all_all_row_maxes = []
        for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split_new(k_bucket_size, axis = -2),
                v.split_new(k_bucket_size, axis = -2),
                row_mask
            )

            for k_ind, (kc, vc, col_mask) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size
                attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                if exists(col_mask):
                    attn_weights.masked_fill_(~col_mask, max_neg_value)

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = paddle.ones((qc.shape[-2], kc.shape[-2]), dtype = paddle.bool,).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                block_row_maxes = attn_weights.amax(axis = -1, keepdim = True)
                new_row_maxes = paddle.maximum(block_row_maxes, row_maxes)

                exp_weights = paddle.exp(attn_weights - new_row_maxes)

                if exists(col_mask):
                    exp_weights.masked_fill_(~col_mask, 0.)

                block_row_sums = exp_weights.sum(axis = -1, keepdim = True).clip(min = EPSILON)

                exp_values = einsum('... i j, ... j d -> ... i d', exp_weights, vc)

                exp_row_max_diff = paddle.exp(row_maxes - new_row_maxes)

                new_row_sums = exp_row_max_diff * row_sums + block_row_sums
                oc.mul_(exp_row_max_diff).add_(exp_values)
                
                row_maxes.copy_(new_row_maxes, False)
                row_sums.copy_(new_row_sums, False)

            oc.div_(row_sums)
            all_all_oc.append(oc)
            all_all_row_sums.append(row_sums)
            all_all_row_maxes.append(row_maxes)

        o = paddle.concat(all_all_oc, axis=-2)
        all_row_sums = paddle.concat(all_all_row_sums, axis=-2)
        all_row_maxes = paddle.concat(all_all_row_maxes, axis=-2)
        
        lse = all_row_sums.log() + all_row_maxes
        
        ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size)
        ctx.save_for_backward(q, k, v, o, lse)

        return o

    @staticmethod
    @paddle.no_grad()
    def backward(ctx, do):
        """ Algorithm 2 in the v2 paper """

        causal, scale, mask, q_bucket_size, k_bucket_size = ctx.args
        q, k, v, o, lse = ctx.saved_tensor()


        max_neg_value = -paddle.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        dq = paddle.zeros_like(q)
        dk = paddle.zeros_like(k)
        dv = paddle.zeros_like(v)

        row_splits = zip(
            q.split_new(q_bucket_size, axis = -2),
            o.split_new(q_bucket_size, axis = -2),
            do.split_new(q_bucket_size, axis = -2),
            mask,
            lse.split_new(q_bucket_size, axis = -2),
            dq.split_new(q_bucket_size, axis = -2)
        )
        all_dqc = []
        all_dkc_val = 0.0
        all_dvc_val = 0.0
        for ind, (qc, oc, doc, row_mask, lsec, dqc) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff
            all_dkc = []
            all_dvc = []
        
            col_splits = zip(
                k.split_new(k_bucket_size, axis = -2),
                v.split_new(k_bucket_size, axis = -2),
                dk.split_new(k_bucket_size, axis = -2),
                dv.split_new(k_bucket_size, axis = -2),
                row_mask
            )
            for k_ind, (kc, vc, dkc, dvc, col_mask) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = paddle.ones((qc.shape[-2], kc.shape[-2]), dtype = paddle.bool,).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                p = paddle.exp(attn_weights - lsec)

                if exists(col_mask):
                    p.masked_fill_(~col_mask, 0.)

                dv_chunk = einsum('... i j, ... i d -> ... j d', p, doc)
                dp = einsum('... i d, ... j d -> ... i j', doc, vc)

                D = (doc * oc).sum(axis = -1, keepdim = True)
                ds = p * scale * (dp - D)

                dq_chunk = einsum('... i j, ... j d -> ... i d', ds, kc)
                dk_chunk = einsum('... i j, ... i d -> ... j d', ds, qc)

                dqc.add_(dq_chunk)
                dkc.add_(dk_chunk)
                dvc.add_(dv_chunk)
                
                all_dkc.append(dkc)
                all_dvc.append(dvc)
                    
            all_dqc.append(dqc)
            all_dkc_val+= paddle.concat(all_dkc, axis=-2)
            all_dvc_val+= paddle.concat(all_dvc, axis=-2)

        dq = paddle.concat(all_dqc, axis = -2)
        return dq, all_dkc_val, all_dvc_val

USE_PYTHON_FLASH_ATTENTION = False

class FlashAttnProcessor:
    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ) -> Any:
        q_bucket_size = 512
        k_bucket_size = 1024

        h = attn.heads
        q = attn.to_q(hidden_states)

        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        encoder_hidden_states = encoder_hidden_states.to(hidden_states.dtype)

        if hasattr(attn, "hypernetwork") and attn.hypernetwork is not None:
            context_k, context_v = attn.hypernetwork.forward(
                hidden_states, encoder_hidden_states
            )
            context_k = context_k.to(hidden_states.dtype)
            context_v = context_v.to(hidden_states.dtype)
        else:
            context_k = encoder_hidden_states
            context_v = encoder_hidden_states

        k = attn.to_k(context_k)
        v = attn.to_v(context_v)
        del encoder_hidden_states, hidden_states

        if USE_PYTHON_FLASH_ATTENTION:
            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
            
            out = FlashAttentionFunction.apply(
                q, k, v, attention_mask, False, q_bucket_size, k_bucket_size
            )

            out = rearrange(out, "b h n d -> b n (h d)")
        else:
            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=h), (q, k, v))
            
            out = paddle.nn.functional.scaled_dot_product_attention_(
                q, k, v, 
            )

            out = rearrange(out, "b n h d -> b n (h d)")

        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out
