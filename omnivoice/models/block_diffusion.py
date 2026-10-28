#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Luna-TTS Realtime block-diffusion helpers.

Luna-TTS Realtime (arXiv:2608.11593) keeps OmniVoice's masked RVQ diffusion
objective and replaces global bidirectional attention with BD3-LM block-causal
attention (Arriola et al., 2025). Training uses the two-stream layout

    [noisy target blocks | clean context + clean target blocks]

so one forward computes every block's denoising loss. Inference prefills the
text/prompt once, denoises one block of ``block_size`` frames in parallel, then
commits that block into the KV cache and streams it.

The speech tokenizer stays the OmniVoice codec. A new audio EOS id is appended
past the mask id so a checkpoint can grow into the realtime vocabulary without
shifting existing codebook rows except at the per-layer boundary.
"""

from __future__ import annotations

import logging
import math
from typing import Iterator, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

CONTEXT_BLOCK_ID = -1
PADDING_BLOCK_ID = -100
STREAM_NOISY = 0
STREAM_CLEAN = 1
STREAM_PAD = -1

# Minimum mask rate. The ELBO weight is 1/t, so t = 0 is undefined.
MIN_MASK_T = 1e-3


def resize_audio_vocab(model, new_vocab_size: int) -> None:
    """Grow per-codebook audio embeddings and heads, preserving old rows.

    OmniVoice packs codebook tables as ``index = layer * vocab + token``.
    Inserting an EOS id therefore cannot be a single ``resize_`` of the
    flattened table: each layer slice has to be copied into the new stride.
    New rows (the EOS id) are initialized from the mask embedding so the
    realtime adaptation does not start from an arbitrary special token.
    """
    old_vocab = model.config.audio_vocab_size
    if new_vocab_size == old_vocab:
        return
    if new_vocab_size < old_vocab:
        raise ValueError(
            f"Refusing to shrink audio vocab from {old_vocab} to {new_vocab_size}."
        )

    num_layers = model.config.num_audio_codebook
    hidden = model.audio_embeddings.embedding_dim
    device = model.audio_embeddings.weight.device
    dtype = model.audio_embeddings.weight.dtype
    mask_id = model.config.audio_mask_id

    new_emb = nn.Embedding(num_layers * new_vocab_size, hidden)
    new_head = nn.Linear(hidden, num_layers * new_vocab_size, bias=False)
    nn.init.normal_(new_emb.weight, std=0.02)
    nn.init.normal_(new_head.weight, std=0.02)

    with torch.no_grad():
        old_emb = model.audio_embeddings.weight.data
        old_head = model.audio_heads.weight.data
        for layer in range(num_layers):
            old_slice = slice(layer * old_vocab, (layer + 1) * old_vocab)
            new_slice = slice(layer * new_vocab_size, layer * new_vocab_size + old_vocab)
            new_emb.weight.data[new_slice] = old_emb[old_slice]
            new_head.weight.data[new_slice] = old_head[old_slice]
            # Copy the mask row into every newly added id (normally just EOS).
            for token_id in range(old_vocab, new_vocab_size):
                src = layer * old_vocab + mask_id
                dst = layer * new_vocab_size + token_id
                new_emb.weight.data[dst] = old_emb[src]
                new_head.weight.data[dst] = old_head[src]

    new_emb.to(device=device, dtype=dtype)
    new_head.to(device=device, dtype=dtype)
    model.audio_embeddings = new_emb
    model.audio_heads = new_head
    model.config.audio_vocab_size = new_vocab_size

    offsets = torch.arange(num_layers, device=device) * new_vocab_size
    model._buffers.pop("codebook_layer_offsets", None)
    model.register_buffer("codebook_layer_offsets", offsets)
    logger.info(
        "Resized audio vocab %d -> %d (%d codebooks).",
        old_vocab,
        new_vocab_size,
        num_layers,
    )


def apply_block_causal_adaptation(model, train_config) -> None:
    """Attach realtime hyperparameters and grow the audio vocabulary if needed."""
    eos_id = int(train_config.audio_eos_id)
    if eos_id == train_config.audio_mask_id:
        raise ValueError("audio_eos_id must be distinct from audio_mask_id.")
    needed = max(int(train_config.audio_vocab_size), eos_id + 1)
    if model.config.audio_vocab_size < needed:
        resize_audio_vocab(model, needed)
    if eos_id >= model.config.audio_vocab_size:
        raise ValueError(
            f"audio_eos_id={eos_id} does not fit vocab {model.config.audio_vocab_size}."
        )
    model.config.audio_eos_id = eos_id
    model.config.block_size = int(train_config.block_size)
    model.config.block_steps = int(train_config.block_steps)
    model.config.audio_mask_id = int(train_config.audio_mask_id)


def build_block_attention_mask(
    block_ids: torch.Tensor,
    stream_ids: torch.Tensor,
    document_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Boolean mask ``[B, 1, L, L]``; True means the query may attend to the key.

    Noisy queries (the xt stream) attend to the current noisy block and to clean
    tokens from strictly earlier blocks, including the text/prompt context.
    Clean queries attend only to clean tokens in blocks up to and including
    their own. Padding and other documents are invisible.
    """
    if block_ids.dim() != 2:
        raise ValueError(f"block_ids must be [B, L], got {tuple(block_ids.shape)}")
    batch, length = block_ids.shape
    if document_ids is None:
        document_ids = torch.arange(batch, device=block_ids.device).unsqueeze(1)
        document_ids = document_ids.expand(batch, length).clone()
        document_ids = document_ids.masked_fill(stream_ids < 0, -1)

    q_block = block_ids[:, :, None]
    k_block = block_ids[:, None, :]
    q_stream = stream_ids[:, :, None]
    k_stream = stream_ids[:, None, :]
    q_doc = document_ids[:, :, None]
    k_doc = document_ids[:, None, :]

    same_doc = (q_doc == k_doc) & (q_doc >= 0) & (k_doc >= 0)
    noisy_query = q_stream == STREAM_NOISY
    noisy_ok = ((k_stream == STREAM_NOISY) & (k_block == q_block)) | (
        (k_stream == STREAM_CLEAN) & (k_block < q_block)
    )
    clean_query = q_stream == STREAM_CLEAN
    clean_ok = (k_stream == STREAM_CLEAN) & (k_block <= q_block)
    allow = same_doc & ((noisy_query & noisy_ok) | (clean_query & clean_ok))
    return allow[:, None]


def build_flex_block_mask(block_ids, stream_ids, document_ids, device):
    """FlexAttention BlockMask for a packed ``[1, L]`` training sequence."""
    from torch.nn.attention.flex_attention import create_block_mask

    docs = document_ids[0].to(device)
    blocks = block_ids[0].to(device)
    streams = stream_ids[0].to(device)

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = (docs[q_idx] == docs[kv_idx]) & (docs[q_idx] >= 0)
        q_stream = streams[q_idx]
        k_stream = streams[kv_idx]
        q_block = blocks[q_idx]
        k_block = blocks[kv_idx]
        noisy_ok = ((k_stream == STREAM_NOISY) & (k_block == q_block)) | (
            (k_stream == STREAM_CLEAN) & (k_block < q_block)
        )
        clean_ok = (k_stream == STREAM_CLEAN) & (k_block <= q_block)
        noisy_query = q_stream == STREAM_NOISY
        clean_query = q_stream == STREAM_CLEAN
        return same_doc & ((noisy_query & noisy_ok) | (clean_query & clean_ok))

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=block_ids.size(-1),
        KV_LEN=block_ids.size(-1),
        _compile=True,
        device=device,
    )


def attention_mask_for_impl(mask: torch.Tensor, attn_impl: str, dtype: torch.dtype):
    """SDPA wants a boolean keep-mask; eager wants an additive float mask."""
    if attn_impl == "eager":
        additive = torch.zeros(mask.shape, device=mask.device, dtype=dtype)
        additive = additive.masked_fill(~mask, torch.finfo(dtype).min)
        return additive
    return mask


def _unmask_counts(num_positions: int, num_step: int, t_shift: float, device) -> list[int]:
    from omnivoice.models.omnivoice import _get_time_steps

    timesteps = _get_time_steps(
        t_start=0.0,
        t_end=1.0,
        num_step=num_step,
        t_shift=t_shift,
        device=device,
    ).tolist()
    remaining = num_positions
    counts = []
    for step in range(num_step):
        if step == num_step - 1:
            n = remaining
        else:
            n = min(
                math.ceil(num_positions * (timesteps[step + 1] - timesteps[step])),
                remaining,
            )
        counts.append(int(n))
        remaining -= int(n)
    return counts


def _forward_block(model, input_ids, audio_mask, cache, *, keep: bool):
    """Run one block (or a prefix) against ``cache``.

    Denoising steps pass ``keep=False`` so the noisy block is cropped back out
    of the cache. The commit step passes ``keep=True``.
    """
    from transformers.cache_utils import DynamicCache

    if cache is None:
        cache = DynamicCache()
    prefix = cache.get_seq_length()
    query_len = input_ids.shape[-1]
    # Zero additive mask: every query may see the cached prefix and the whole
    # current block, and `attention_mask is not None` disables the backbone's
    # causal flag. Future blocks are simply not in the cache yet.
    attn = torch.zeros(
        input_ids.shape[0],
        1,
        query_len,
        prefix + query_len,
        dtype=next(model.parameters()).dtype,
        device=input_ids.device,
    )
    outputs = model(
        input_ids=input_ids,
        audio_mask=audio_mask,
        attention_mask=attn,
        past_key_values=cache,
        use_cache=True,
    )
    if not keep:
        cache.crop(prefix)
    return outputs.logits, cache


def _tokenize_prefix(model, text, ref_text, lang, instruct, denoise, has_prompt):
    from omnivoice.models.omnivoice import _combine_text, _tokenize_with_nonverbal_tags

    channels = model.config.num_audio_codebook
    style = ""
    if denoise and has_prompt:
        style += "<|denoise|>"
    style += f"<|lang_start|>{lang or 'None'}<|lang_end|>"
    style += f"<|instruct_start|>{instruct or 'None'}<|instruct_end|>"
    style_ids = (
        model.text_tokenizer(style, return_tensors="pt")
        .input_ids.repeat(channels, 1)
        .unsqueeze(0)
    )
    wrapped = f"<|text_start|>{_combine_text(text, ref_text)}<|text_end|>"
    text_ids = (
        _tokenize_with_nonverbal_tags(wrapped, model.text_tokenizer)
        .repeat(channels, 1)
        .unsqueeze(0)
    )
    return style_ids, text_ids


def _commit_schedule_step(tokens, pred_tokens, scores, k, mask_id, position_temperature):
    from omnivoice.models.omnivoice import _gumbel_sample

    if position_temperature > 0.0:
        scores = _gumbel_sample(scores, position_temperature)
    scores = scores.masked_fill(tokens != mask_id, float("-inf"))
    k = min(k, int((tokens == mask_id).sum().item()))
    if k <= 0:
        return tokens
    _, topk_idx = torch.topk(scores.flatten(), k)
    flat = tokens.flatten()
    flat[topk_idx] = pred_tokens.flatten()[topk_idx]
    return flat.view_as(tokens)


def iter_realtime_blocks(
    model,
    *,
    text: str,
    ref_text: Optional[str],
    ref_audio_tokens: Optional[torch.Tensor],
    lang: Optional[str],
    instruct: Optional[str],
    gen_config,
    max_duration: float,
    frame_rate: float,
) -> Iterator[torch.Tensor]:
    """Yield each committed speech block as ``[C, T]`` tokens, EOS stripped.

    Stops on the learned EOS frame or on ``max_duration``, whichever comes first.
    """
    from transformers.cache_utils import DynamicCache

    block_size = int(getattr(model.config, "block_size", 0) or 0)
    eos_id = getattr(model.config, "audio_eos_id", None)
    if block_size <= 0 or eos_id is None:
        raise RuntimeError(
            "This checkpoint is not a Luna-TTS Realtime model. Train with "
            "block_causal=true (see examples/config/train_config_luna_realtime.json) "
            "so config.block_size and config.audio_eos_id are set."
        )

    device = model.device
    channels = model.config.num_audio_codebook
    mask_id = model.config.audio_mask_id
    has_prompt = ref_audio_tokens is not None and ref_audio_tokens.numel() > 0

    style_ids, text_ids = _tokenize_prefix(
        model,
        text,
        ref_text,
        lang,
        instruct,
        gen_config.denoise,
        has_prompt,
    )
    cond_parts = [style_ids.to(device), text_ids.to(device)]
    uncond_parts = []
    if has_prompt:
        prompt = ref_audio_tokens.unsqueeze(0).to(device=device, dtype=torch.long)
        cond_parts.append(prompt)
        uncond_parts.append(prompt)
    cond_prefix = torch.cat(cond_parts, dim=2).to(device)
    cond_audio = torch.zeros(
        1, cond_prefix.shape[-1], dtype=torch.bool, device=device
    )
    # Style and text occupy the non-audio prefix; prompt, if any, is the tail.
    prompt_len = ref_audio_tokens.shape[-1] if has_prompt else 0
    if prompt_len:
        cond_audio[0, -prompt_len:] = True

    cond_cache = DynamicCache()
    _, cond_cache = _forward_block(
        model, cond_prefix, cond_audio, cond_cache, keep=True
    )

    use_cfg = gen_config.guidance_scale != 0
    uncond_cache = None
    if use_cfg:
        uncond_cache = DynamicCache()
        if uncond_parts:
            uncond_prefix = torch.cat(uncond_parts, dim=2).to(device)
            uncond_audio = torch.ones(
                1, uncond_prefix.shape[-1], dtype=torch.bool, device=device
            )
            _, uncond_cache = _forward_block(
                model, uncond_prefix, uncond_audio, uncond_cache, keep=True
            )

    max_frames = max(block_size, int(max_duration * frame_rate))
    max_blocks = math.ceil(max_frames / block_size)
    produced = 0
    layer_ids = torch.arange(channels, device=device).view(1, -1, 1)
    positions = block_size * channels

    for _ in range(max_blocks):
        tokens = torch.full(
            (1, channels, block_size),
            mask_id,
            dtype=torch.long,
            device=device,
        )
        audio_mask = torch.ones(1, block_size, dtype=torch.bool, device=device)
        counts = _unmask_counts(
            positions, gen_config.num_step, gen_config.t_shift, device
        )
        for step, k in enumerate(counts):
            cond_logits, cond_cache = _forward_block(
                model, tokens, audio_mask, cond_cache, keep=False
            )
            if use_cfg:
                uncond_logits, uncond_cache = _forward_block(
                    model, tokens, audio_mask, uncond_cache, keep=False
                )
            else:
                uncond_logits = cond_logits
            pred, scores = model._predict_tokens_with_scoring(
                cond_logits[:, :, :block_size],
                uncond_logits[:, :, :block_size],
                gen_config,
            )
            scores = scores - layer_ids * gen_config.layer_penalty_factor
            tokens = _commit_schedule_step(
                tokens,
                pred,
                scores,
                k,
                mask_id,
                gen_config.position_temperature,
            )
            if step == gen_config.num_step - 1:
                # The last step of the cosine schedule commits every remaining
                # mask. Force anything still masked (numerical ties) to the argmax.
                still = tokens == mask_id
                tokens = torch.where(still, pred, tokens)

        # Commit the clean block into both caches before reading the next one.
        _, cond_cache = _forward_block(
            model, tokens, audio_mask, cond_cache, keep=True
        )
        if use_cfg:
            _, uncond_cache = _forward_block(
                model, tokens, audio_mask, uncond_cache, keep=True
            )

        block = tokens[0]
        # EOS is trained as a full frame, but a partially-EOS frame can appear
        # during sampling (e.g. channel 0 is speech, channel 3 is EOS). Feeding
        # the EOS/mask ids to the audio tokenizer is out of range and crashes
        # the CUDA context, so stop at the first column containing either.
        stop_mask = (block == eos_id) | (block == mask_id)
        if bool(stop_mask.any()):
            cut = int(stop_mask.any(dim=0).nonzero()[0])
            if cut > 0:
                speech = block[:, :cut]
                produced += speech.shape[-1]
                yield speech
            return
        produced += block_size
        yield block
        if produced >= max_frames:
            logger.warning(
                "Hit max_duration=%.1fs before EOS; truncating.", max_duration
            )
            return
