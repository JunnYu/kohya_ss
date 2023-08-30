# Diffusersのコードをベースとした sd_xl_baseのU-Net
# state dictの形式をSDXLに合わせてある

"""
      target: sgm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        adm_in_channels: 2816
        num_classes: sequential
        use_checkpoint: True
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [4, 2]
        num_res_blocks: 2
        channel_mult: [1, 2, 4]
        num_head_channels: 64
        use_spatial_transformer: True
        use_linear_in_transformer: True
        transformer_depth: [1, 2, 10]  # note: the first is unused (due to attn_res starting at 2) 32, 16, 8 --> 64, 32, 16
        context_dim: 2048
        spatial_transformer_attn_type: softmax-xformers
        legacy: False
"""

import math
from types import SimpleNamespace
from typing import Optional
import paddle
from paddle import nn
from paddle.nn import functional as F
from einops import rearrange


IN_CHANNELS: int = 4
OUT_CHANNELS: int = 4
ADM_IN_CHANNELS: int = 2816
CONTEXT_DIM: int = 2048
MODEL_CHANNELS: int = 320
TIME_EMBED_DIM = 320 * 4


# region memory effcient attention

# FlashAttentionを使うCrossAttention
# based on https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/memory_efficient_attention_pytorch/flash_attention.py
# LICENSE MIT https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/LICENSE

# constants

EPSILON = 1e-6

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# flash attention forwards and backwards

# https://arxiv.org/abs/2205.14135

from .attention_processors import FlashAttentionFunction


# endregion


def get_parameter_dtype(parameter: paddle.nn.Layer):
    return next(parameter.named_parameters())[1].dtype


def get_parameter_device(parameter: paddle.nn.Layer):
    return next(parameter.named_parameters())[1].place



def get_timestep_embedding(
    timesteps: paddle.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * paddle.arange(start=0, end=half_dim, dtype=paddle.float32, )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = paddle.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = paddle.concat([paddle.sin(emb), paddle.cos(emb)], axis=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = paddle.concat([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = paddle.concat(emb, paddle.zeros([emb.shape[0], 1]), axis=-1)
    return emb


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        if self.weight.dtype != paddle.float32:
            return super().forward(x)
        return super().forward(x.float()).to(x.dtype)


class ResnetBlock2D(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_layers = nn.Sequential(
            GroupNorm32(32, in_channels),
            nn.Silu(),
            nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.emb_layers = nn.Sequential(nn.Silu(), nn.Linear(TIME_EMBED_DIM, out_channels))

        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels),
            nn.Silu(),
            nn.Identity(),  # to make state_dict compatible with original model
            nn.Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        if in_channels != out_channels:
            self.skip_connection = nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.skip_connection = nn.Identity()

        self.gradient_checkpointing = False

    def forward_body(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).to(h.dtype)
        h = h + emb_out[:, :, None, None]
        h = self.out_layers(h)
        x = self.skip_connection(x)
        return x + h

    def forward(self, x, emb):
        if self.training and self.gradient_checkpointing:
            # print("ResnetBlock2D: gradient_checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            x = paddle.distributed.fleet.utils.recompute(create_custom_forward(self.forward_body), x, emb)
        else:
            x = self.forward_body(x, emb)

        return x


class Downsample2D(nn.Layer):
    def __init__(self, channels, out_channels):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels

        self.op = nn.Conv2D(self.channels, self.out_channels, 3, stride=2, padding=1)

        self.gradient_checkpointing = False

    def forward_body(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.op(hidden_states)

        return hidden_states

    def forward(self, hidden_states):
        if self.training and self.gradient_checkpointing:
            # print("Downsample2D: gradient_checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            hidden_states = paddle.distributed.fleet.utils.recompute(create_custom_forward(self.forward_body), hidden_states)
        else:
            hidden_states = self.forward_body(hidden_states)

        return hidden_states


class CrossAttention(nn.Layer):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        upcast_attention: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias_attr=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias_attr=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias_attr=False)

        self.to_out = nn.LayerList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        # no dropout here

        self.use_memory_efficient_attention_xformers = False
        self.use_memory_efficient_attention_mem_eff = False
        self.use_sdpa = False

    def set_use_memory_efficient_attention(self, xformers, mem_eff):
        self.use_memory_efficient_attention_xformers = xformers
        self.use_memory_efficient_attention_mem_eff = mem_eff

    def set_use_sdpa(self, sdpa):
        self.use_sdpa = sdpa

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape([batch_size, seq_len, head_size, dim // head_size])
        tensor = tensor.transpose([0, 2, 1, 3]).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape([batch_size // head_size, head_size, seq_len, dim])
        tensor = tensor.transpose([0, 2, 1, 3]).reshape([batch_size // head_size, seq_len, dim * head_size])
        return tensor

    def forward(self, hidden_states, context=None, mask=None):
        if self.use_memory_efficient_attention_xformers:
            return self.forward_memory_efficient_xformers(hidden_states, context, mask)
        if self.use_memory_efficient_attention_mem_eff:
            return self.forward_memory_efficient_mem_eff(hidden_states, context, mask)
        if self.use_sdpa:
            return self.forward_sdpa(hidden_states, context, mask)

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        hidden_states = self._attention(query, key, value)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # hidden_states = self.to_out[1](hidden_states)     # no dropout
        return hidden_states

    def _attention(self, query, key, value):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = paddle.matmul(query, key, transpose_y=True) * self.scale

        attention_probs = F.softmax(attention_scores, axis=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = paddle.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    # TODO support Hypernetworks
    def forward_memory_efficient_xformers(self, x, context=None, mask=None):
        import xformers.ops

        h = self.heads
        q_in = self.to_q(x)
        context = context if context is not None else x
        context = context.to(x.dtype)
        k_in = self.to_k(context)
        v_in = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        out = F.scaled_dot_product_attention_(q, k, v, attn_mask=None, scale=self.scale)  # 最適なのを選んでくれる
        del q, k, v

        out = rearrange(out, "b n h d -> b n (h d)", h=h)

        out = self.to_out[0](out)
        return out

    def forward_memory_efficient_mem_eff(self, x, context=None, mask=None):
        flash_func = FlashAttentionFunction

        q_bucket_size = 512
        k_bucket_size = 1024

        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x
        context = context.to(x.dtype)
        k = self.to_k(context)
        v = self.to_v(context)
        del context, x

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        out = flash_func.apply(q, k, v, mask, False, q_bucket_size, k_bucket_size)

        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.to_out[0](out)
        return out

    def forward_sdpa(self, x, context=None, mask=None):
        h = self.heads
        q_in = self.to_q(x)
        context = context if context is not None else x
        context = context.to(x.dtype)
        k_in = self.to_k(context)
        v_in = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        out = F.scaled_dot_product_attention_(q, k, v, attn_mask=mask, )

        out = rearrange(out, "b n h d -> b n (h d)", h=h)

        out = self.to_out[0](out)
        return out


# feedforward
class GEGLU(nn.Layer):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=paddle.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, axis=-1)
        return hidden_states * self.gelu(gate)


class FeedForward(nn.Layer):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        inner_dim = int(dim * 4)  # mult is always 4

        self.net = nn.LayerList([])
        # project in
        self.net.append(GEGLU(dim, inner_dim))
        # project dropout
        self.net.append(nn.Identity())  # nn.Dropout(0)) # dummy for dropout with 0
        # project out
        self.net.append(nn.Linear(inner_dim, dim))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class BasicTransformerBlock(nn.Layer):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, cross_attention_dim: int, upcast_attention: bool = False
    ):
        super().__init__()

        self.gradient_checkpointing = False

        # 1. Self-Attn
        self.attn1 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            upcast_attention=upcast_attention,
        )
        self.ff = FeedForward(dim)

        # 2. Cross-Attn
        self.attn2 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            upcast_attention=upcast_attention,
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim)

    def set_use_memory_efficient_attention(self, xformers: bool, mem_eff: bool):
        self.attn1.set_use_memory_efficient_attention(xformers, mem_eff)
        self.attn2.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, sdpa: bool):
        self.attn1.set_use_sdpa(sdpa)
        self.attn2.set_use_sdpa(sdpa)

    def forward_body(self, hidden_states, context=None, timestep=None):
        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)

        hidden_states = self.attn1(norm_hidden_states) + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        hidden_states = self.attn2(norm_hidden_states, context=context) + hidden_states

        # 3. Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states

    def forward(self, hidden_states, context=None, timestep=None):
        if self.training and self.gradient_checkpointing:
            # print("BasicTransformerBlock: checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            output = paddle.distributed.fleet.utils.recompute(create_custom_forward(self.forward_body), hidden_states, context, timestep)
        else:
            output = self.forward_body(hidden_states, context, timestep)

        return output


class Transformer2DModel(nn.Layer):
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        use_linear_projection: bool = False,
        upcast_attention: bool = False,
        num_transformer_layers: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.use_linear_projection = use_linear_projection

        self.norm = paddle.nn.GroupNorm(num_groups=32, num_channels=in_channels, epsilon=1e-6,)
        # self.norm = GroupNorm32(32, in_channels, eps=1e-6, affine=True)

        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2D(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        blocks = []
        for _ in range(num_transformer_layers):
            blocks.append(
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                )
            )

        self.transformer_blocks = nn.LayerList(blocks)

        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2D(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

        self.gradient_checkpointing = False

    def set_use_memory_efficient_attention(self, xformers, mem_eff):
        for transformer in self.transformer_blocks:
            transformer.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, sdpa):
        for transformer in self.transformer_blocks:
            transformer.set_use_sdpa(sdpa)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None):
        # 1. Input
        batch, _, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.transpose([0, 2, 3, 1]).reshape([batch, height * weight, inner_dim])
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.transpose([0, 2, 3, 1]).reshape([batch, height * weight, inner_dim])
            hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context=encoder_hidden_states, timestep=timestep)

        # 3. Output
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape([batch, height, weight, inner_dim]).transpose([0, 3, 1, 2])
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape([batch, height, weight, inner_dim]).transpose([0, 3, 1, 2])

        output = hidden_states + residual

        return output


class Upsample2D(nn.Layer):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.conv = nn.Conv2D(self.channels, self.out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward_body(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == paddle.bfloat16:
            hidden_states = hidden_states.to(paddle.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        # if hidden_states.shape[0] >= 64:
        #     hidden_states = hidden_states

        # if `output_size` is passed we force the interpolation output size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == paddle.bfloat16:
            hidden_states = hidden_states.to(dtype)

        hidden_states = self.conv(hidden_states)

        return hidden_states

    def forward(self, hidden_states, output_size=None):
        if self.training and self.gradient_checkpointing:
            # print("Upsample2D: gradient_checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            hidden_states = paddle.distributed.fleet.utils.recompute(create_custom_forward(self.forward_body), hidden_states, output_size)
        else:
            hidden_states = self.forward_body(hidden_states, output_size)

        return hidden_states


class SdxlUNet2DConditionModel(nn.Layer):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = IN_CHANNELS
        self.out_channels = OUT_CHANNELS
        self.model_channels = MODEL_CHANNELS
        self.time_embed_dim = TIME_EMBED_DIM
        self.adm_in_channels = ADM_IN_CHANNELS

        self.gradient_checkpointing = False
        # self.sample_size = sample_size

        # time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, self.time_embed_dim),
            nn.Silu(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # label embedding
        self.label_emb = nn.Sequential(
            nn.Sequential(
                nn.Linear(self.adm_in_channels, self.time_embed_dim),
                nn.Silu(),
                nn.Linear(self.time_embed_dim, self.time_embed_dim),
            )
        )

        # input
        self.input_blocks = nn.LayerList(
            [
                nn.Sequential(
                    nn.Conv2D(self.in_channels, self.model_channels, kernel_size=3, padding=(1, 1)),
                )
            ]
        )

        # level 0
        for i in range(2):
            layers = [
                ResnetBlock2D(
                    in_channels=1 * self.model_channels,
                    out_channels=1 * self.model_channels,
                ),
            ]
            self.input_blocks.append(nn.LayerList(layers))

        self.input_blocks.append(
            nn.Sequential(
                Downsample2D(
                    channels=1 * self.model_channels,
                    out_channels=1 * self.model_channels,
                ),
            )
        )

        # level 1
        for i in range(2):
            layers = [
                ResnetBlock2D(
                    in_channels=(1 if i == 0 else 2) * self.model_channels,
                    out_channels=2 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=2 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=2 * self.model_channels,
                    num_transformer_layers=2,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
            ]
            self.input_blocks.append(nn.LayerList(layers))

        self.input_blocks.append(
            nn.Sequential(
                Downsample2D(
                    channels=2 * self.model_channels,
                    out_channels=2 * self.model_channels,
                ),
            )
        )

        # level 2
        for i in range(2):
            layers = [
                ResnetBlock2D(
                    in_channels=(2 if i == 0 else 4) * self.model_channels,
                    out_channels=4 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=4 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=4 * self.model_channels,
                    num_transformer_layers=10,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
            ]
            self.input_blocks.append(nn.LayerList(layers))

        # mid
        self.middle_block = nn.LayerList(
            [
                ResnetBlock2D(
                    in_channels=4 * self.model_channels,
                    out_channels=4 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=4 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=4 * self.model_channels,
                    num_transformer_layers=10,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
                ResnetBlock2D(
                    in_channels=4 * self.model_channels,
                    out_channels=4 * self.model_channels,
                ),
            ]
        )

        # output
        self.output_blocks = nn.LayerList([])

        # level 2
        for i in range(3):
            layers = [
                ResnetBlock2D(
                    in_channels=4 * self.model_channels + (4 if i <= 1 else 2) * self.model_channels,
                    out_channels=4 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=4 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=4 * self.model_channels,
                    num_transformer_layers=10,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
            ]
            if i == 2:
                layers.append(
                    Upsample2D(
                        channels=4 * self.model_channels,
                        out_channels=4 * self.model_channels,
                    )
                )

            self.output_blocks.append(nn.LayerList(layers))

        # level 1
        for i in range(3):
            layers = [
                ResnetBlock2D(
                    in_channels=2 * self.model_channels + (4 if i == 0 else (2 if i == 1 else 1)) * self.model_channels,
                    out_channels=2 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=2 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=2 * self.model_channels,
                    num_transformer_layers=2,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
            ]
            if i == 2:
                layers.append(
                    Upsample2D(
                        channels=2 * self.model_channels,
                        out_channels=2 * self.model_channels,
                    )
                )

            self.output_blocks.append(nn.LayerList(layers))

        # level 0
        for i in range(3):
            layers = [
                ResnetBlock2D(
                    in_channels=1 * self.model_channels + (2 if i == 0 else 1) * self.model_channels,
                    out_channels=1 * self.model_channels,
                ),
            ]

            self.output_blocks.append(nn.LayerList(layers))

        # output
        self.out = nn.LayerList(
            [GroupNorm32(32, self.model_channels), nn.Silu(), nn.Conv2D(self.model_channels, self.out_channels, 3, padding=1)]
        )

    # region diffusers compatibility
    def prepare_config(self):
        self.config = SimpleNamespace()

    @property
    def dtype(self) -> paddle.dtype:
        # `paddle.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        return get_parameter_dtype(self)

    @property
    def device(self) -> str:
        # `torch.device`: The device on which the module is (assuming that all the module parameters are on the same device).
        return get_parameter_device(self)

    def set_attention_slice(self, slice_size):
        raise NotImplementedError("Attention slicing is not supported for this model.")

    def is_gradient_checkpointing(self) -> bool:
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.sublayers())

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        self.set_gradient_checkpointing(value=True)

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.set_gradient_checkpointing(value=False)

    def set_use_memory_efficient_attention(self, xformers: bool, mem_eff: bool) -> None:
        # blocks = list(self.input_blocks) + [self.middle_block] + list(self.output_blocks)
        # for block in blocks:
        for module in self.sublayers(include_self=False):
            if hasattr(module, "set_use_memory_efficient_attention"):
                # print(module.__class__.__name__)
                module.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, sdpa: bool) -> None:
        # blocks = list(self.input_blocks) + [self.middle_block] + list(self.output_blocks)
        # for block in blocks:
        for module in self.sublayers(include_self=False):
            if hasattr(module, "set_use_sdpa"):
                module.set_use_sdpa(sdpa)

    def set_gradient_checkpointing(self, value=False):
        # blocks = list(self.input_blocks) + [self.middle_block] + list(self.output_blocks)
        # for block in blocks:
        for module in self.sublayers(include_self=False):
            if hasattr(module, "gradient_checkpointing"):
                # print(module.__class__.__name__, module.gradient_checkpointing, "->", value)
                module.gradient_checkpointing = value

    # endregion

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        # broadcast timesteps to batch dimension
        timesteps = timesteps.expand((x.shape[0],))

        hs = []
        t_emb = get_timestep_embedding(timesteps, self.model_channels)  # , repeat_only=False)
        t_emb = t_emb.to(x.dtype)
        emb = self.time_embed(t_emb)

        assert x.shape[0] == y.shape[0], f"batch size mismatch: {x.shape[0]} != {y.shape[0]}"
        assert x.dtype == y.dtype, f"dtype mismatch: {x.dtype} != {y.dtype}"
        # assert x.dtype == self.dtype
        emb = emb + self.label_emb(y)

        def call_module(module, h, emb, context):
            x = h
            for layer in module:
                # print(layer.__class__.__name__, x.dtype, emb.dtype, context.dtype if context is not None else None)
                if isinstance(layer, ResnetBlock2D):
                    x = layer(x, emb)
                elif isinstance(layer, Transformer2DModel):
                    x = layer(x, context)
                else:
                    x = layer(x)
            return x

        # h = x.type(self.dtype)
        h = x
        for module in self.input_blocks:
            h = call_module(module, h, emb, context)
            hs.append(h)

        h = call_module(self.middle_block, h, emb, context)

        for module in self.output_blocks:
            h = paddle.concat([h, hs.pop()], axis=1)
            h = call_module(module, h, emb, context)

        h = h.to(x.dtype)
        h = call_module(self.out, h, emb, context)

        return h


# if __name__ == "__main__":
#     import time

#     print("create unet")
#     unet = SdxlUNet2DConditionModel()

#     unet.set_use_sdpa(True,)
#     unet.set_gradient_checkpointing(True)
#     unet.train()

#     # 使用メモリ量確認用の疑似学習ループ
#     print("preparing optimizer")

#     # optimizer = torch.optim.SGD(unet.parameters(), lr=1e-3, nesterov=True, momentum=0.9) # not working

#     # import bitsandbytes
#     # optimizer = bitsandbytes.adam.Adam8bit(unet.parameters(), lr=1e-3)        # not working
#     # optimizer = bitsandbytes.optim.RMSprop8bit(unet.parameters(), lr=1e-3)  # working at 23.5 GB with torch2
#     # optimizer=bitsandbytes.optim.Adagrad8bit(unet.parameters(), lr=1e-3)  # working at 23.5 GB with torch2

#     from paddle.optimizer import AdamW
#     optimizer = AdamW(parameters=unet.parameters(), learning_rate=1e-3)  # working at 22.2GB with torch2
#     import paddle.amp
#     scaler = paddle.amp.GradScaler(enable=True)

#     print("start training")
#     steps = 10
#     batch_size = 1

#     for step in range(steps):
#         print(f"step {step}")
#         if step == 1:
#             time_start = time.perf_counter()

#         x = paddle.randn([batch_size, 4, 128, 128]) # 1024x1024
#         t = paddle.randint(low=0, high=10, shape=(batch_size,),)
#         ctx = paddle.randn([batch_size, 77, 2048])
#         y = paddle.randn([batch_size, ADM_IN_CHANNELS])

#         with paddle.amp.auto_cast(enable=True):
#             output = unet(x, t, ctx, y)
#             target = paddle.randn_like(output)
#             loss = paddle.nn.functional.mse_loss(output, target)

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         optimizer.clear_grad()

#     time_end = time.perf_counter()
#     print(f"elapsed time: {time_end - time_start} [sec] for last {steps - 1} steps")
