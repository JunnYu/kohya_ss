import paddle
import paddle.nn.functional as F
from ppdiffusers.models.attention_processor import (
    Attention,
    AttnProcessor2_5,
    SlicedAttnProcessor,
    XFormersAttnProcessor
)

xformers = None


loaded_networks = []


def apply_single_hypernetwork(
    hypernetwork, hidden_states, encoder_hidden_states
):
    context_k, context_v = hypernetwork.forward(hidden_states, encoder_hidden_states)
    return context_k, context_v


def apply_hypernetworks(context_k, context_v, layer=None):
    if len(loaded_networks) == 0:
        return context_v, context_v
    for hypernetwork in loaded_networks:
        context_k, context_v = hypernetwork.forward(context_k, context_v)

    context_k = context_k.to(dtype=context_k.dtype)
    context_v = context_v.to(dtype=context_k.dtype)

    return context_k, context_v



def xformers_forward(
    self: XFormersAttnProcessor,
    attn: Attention,
    hidden_states: paddle.Tensor,
    encoder_hidden_states: paddle.Tensor = None,
    attention_mask: paddle.Tensor = None,
):
    batch_size, sequence_length, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )

    attention_mask = attn.prepare_attention_mask(
        attention_mask, sequence_length, batch_size
    )

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    context_k, context_v = apply_hypernetworks(hidden_states, encoder_hidden_states)

    key = attn.to_k(context_k)
    value = attn.to_v(context_v)

    query = attn.head_to_batch_dim(query, transpose=False)
    key = attn.head_to_batch_dim(key, transpose=False)
    value = attn.head_to_batch_dim(value, transpose=False)

    hidden_states = F.scaled_dot_product_attention_(
        query,
        key,
        value,
        attn_mask=attention_mask,
        scale=attn.scale,
        dropout_p=0.0,
        training=attn.training,
        attention_op=self.attention_op, )

    hidden_states = hidden_states.to(query.dtype)
    hidden_states = attn.batch_to_head_dim(hidden_states, transpose=False)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states


def sliced_attn_forward(
    self: SlicedAttnProcessor,
    attn: Attention,
    hidden_states: paddle.Tensor,
    encoder_hidden_states: paddle.Tensor = None,
    attention_mask: paddle.Tensor = None,
):
    batch_size, sequence_length, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )
    attention_mask = attn.prepare_attention_mask(
        attention_mask, sequence_length, batch_size
    )

    query = attn.to_q(hidden_states)
    dim = query.shape[-1]
    query = attn.head_to_batch_dim(query)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    context_k, context_v = apply_hypernetworks(hidden_states, encoder_hidden_states)

    key = attn.to_k(context_k)
    value = attn.to_v(context_v)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    query = query.flatten(0, 1)
    key = key.flatten(0, 1)
    value = value.flatten(0, 1)
        
    batch_size_attention = query.shape[0]
    query_len = query.shape[1]
    hidden_states = paddle.zeros(
        (batch_size_attention, query_len, attn.head_dim), dtype=query.dtype)

    for i in range(batch_size_attention // self.slice_size):
        start_idx = i * self.slice_size
        end_idx = (i + 1) * self.slice_size

        query_slice = query[start_idx:end_idx]
        key_slice = key[start_idx:end_idx]
        attn_mask_slice = attention_mask[
            start_idx:end_idx] if attention_mask is not None else None

        attn_slice = attn.get_attention_scores(query_slice, key_slice,
                                                attn_mask_slice)

        attn_slice = paddle.matmul(attn_slice, value[start_idx:end_idx])

        hidden_states[start_idx:end_idx] = attn_slice

    # reshape back to [bs, num_heads, seqlen, head_dim]
    hidden_states = hidden_states.reshape(
        [-1, attn.heads, query_len, attn.head_dim])

    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    return hidden_states


def v2_5_forward(
    self: AttnProcessor2_5,
    attn: Attention,
    hidden_states,
    encoder_hidden_states=None,
    attention_mask=None,
):
    batch_size, sequence_length, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.view(
            batch_size, attn.heads, -1, attention_mask.shape[-1]
        )

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    context_k, context_v = apply_hypernetworks(hidden_states, encoder_hidden_states)

    key = attn.to_k(context_k)
    value = attn.to_v(context_v)

    query = attn.head_to_batch_dim(query, transpose=False)
    key = attn.head_to_batch_dim(key, transpose=False)
    value = attn.head_to_batch_dim(value, transpose=False)

    hidden_states = F.scaled_dot_product_attention_(
        query,
        key,
        value,
        attn_mask=attention_mask,
        scale=attn.scale,
        dropout_p=0.0,
        training=attn.training,
        attention_op=self.attention_op, )

    hidden_states = hidden_states.to(query.dtype)
    hidden_states = attn.batch_to_head_dim(hidden_states, transpose=False)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states


def replace_attentions_for_hypernetwork():
    import ppdiffusers.models.attention_processor

    ppdiffusers.models.attention_processor.XFormersAttnProcessor.__call__ = (
        xformers_forward
    )
    ppdiffusers.models.attention_processor.SlicedAttnProcessor.__call__ = (
        sliced_attn_forward
    )
    ppdiffusers.models.attention_processor.AttnProcessor2_5.__call__ = v2_5_forward
