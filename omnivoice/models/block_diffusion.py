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

# Compiled realtime runners keyed by (model, device, shape). Keeping them alive
# lets every request after the first reuse the same CUDA graph.
_COMPILED_RUNNERS = {}


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


def _commit_schedule_step(
    tokens,
    pred_tokens,
    scores,
    k,
    mask_id,
    position_temperature,
    remaining: Optional[int] = None,
):
    from omnivoice.models.omnivoice import _gumbel_sample

    if position_temperature > 0.0:
        scores = _gumbel_sample(scores, position_temperature)
    scores = scores.masked_fill(tokens != mask_id, float("-inf"))
    if remaining is None:
        # Kept for callers that do not track the schedule; this forces a GPU
        # sync, so the realtime loop passes the remaining count explicitly.
        remaining = int((tokens == mask_id).sum().item())
    k = min(k, remaining)
    if k <= 0:
        return tokens
    _, topk_idx = torch.topk(scores.flatten(), k)
    flat = tokens.flatten()
    flat[topk_idx] = pred_tokens.flatten()[topk_idx]
    return flat.view_as(tokens)


class _SeparateCFGRunner:
    """Baseline runner: one forward per condition stream (the original path)."""

    def __init__(
        self,
        model,
        *,
        cond_prefix,
        cond_audio,
        uncond_prefix,
        uncond_audio,
        use_cfg: bool,
        block_size: int,
        device,
    ):
        from transformers.cache_utils import DynamicCache

        self.model = model
        self.block_size = block_size
        self.use_cfg = use_cfg
        self.audio_mask = torch.ones(1, block_size, dtype=torch.bool, device=device)

        self.cond_cache = DynamicCache()
        _, self.cond_cache = _forward_block(
            model, cond_prefix, cond_audio, self.cond_cache, keep=True
        )
        self.uncond_cache = None
        if use_cfg:
            self.uncond_cache = DynamicCache()
            if uncond_prefix.shape[-1] > 0:
                _, self.uncond_cache = _forward_block(
                    model, uncond_prefix, uncond_audio, self.uncond_cache, keep=True
                )

    def begin_block(self, _block_index: int) -> None:
        pass

    def step(self, tokens):
        cond_logits, self.cond_cache = _forward_block(
            self.model, tokens, self.audio_mask, self.cond_cache, keep=False
        )
        if self.use_cfg:
            uncond_logits, self.uncond_cache = _forward_block(
                self.model, tokens, self.audio_mask, self.uncond_cache, keep=False
            )
        else:
            uncond_logits = cond_logits
        return cond_logits, uncond_logits

    def commit(self, tokens) -> None:
        _, self.cond_cache = _forward_block(
            self.model, tokens, self.audio_mask, self.cond_cache, keep=True
        )
        if self.use_cfg:
            _, self.uncond_cache = _forward_block(
                self.model, tokens, self.audio_mask, self.uncond_cache, keep=True
            )


class _BatchedCFGRunner:
    """Run cond and uncond as one batch=2 forward on a preallocated cache.

    Two things make this much faster than :class:`_SeparateCFGRunner`:

    * CFG is one batch=2 forward instead of two batch=1 forwards. At these
      sequence lengths the model is launch-bound, so batch=2 costs almost the
      same as batch=1.
    * ``torch.compile(mode="reduce-overhead")`` can capture the fixed-shape
      step (including the KV cache writes) as a CUDA graph. The eager step
      dispatches ~1.7k tiny kernels and leaves the GPU idle; graph replay
      removes that overhead.

    The two condition streams have different prefix lengths, so both are
    right-padded to a common length and the real positions are passed
    explicitly. Padded keys are masked out; RoPE uses the unpadded length.
    """

    def __init__(
        self,
        model,
        *,
        cond_prefix,
        cond_audio,
        uncond_prefix,
        uncond_audio,
        block_size: int,
        max_blocks: int,
        compile_steps: bool,
        device,
        prefix_len: Optional[int] = None,
    ):
        from transformers.cache_utils import StaticCache

        channels = cond_prefix.shape[1]
        natural_prefix = max(cond_prefix.shape[-1], uncond_prefix.shape[-1])
        if prefix_len is None:
            prefix_len = natural_prefix
            if compile_steps:
                # Bucket the prefix length so a whole evaluation run only builds
                # a handful of CUDA graphs, instead of one per sentence.
                prefix_len = max(256, math.ceil(prefix_len / 256) * 256)
        if natural_prefix > prefix_len:
            raise ValueError(
                f"prefix_len={prefix_len} cannot hold {natural_prefix} tokens."
            )
        self.prefix_len = prefix_len
        self.lc = cond_prefix.shape[-1]
        self.lu = uncond_prefix.shape[-1]
        self.block_size = block_size
        self.model = model
        self.compiled = bool(compile_steps)
        self._warmed = False

        max_cache_len = prefix_len + max_blocks * block_size + block_size
        self.cache = StaticCache(
            config=model.llm.config,
            max_cache_len=max_cache_len,
            max_batch_size=2,
        )
        self.ids = torch.zeros(2, channels, prefix_len, dtype=torch.long, device=device)
        self.audio = torch.zeros(2, prefix_len, dtype=torch.bool, device=device)

        mask_dtype = next(model.parameters()).dtype
        neg = torch.finfo(mask_dtype).min
        self.neg = neg
        self.prefill_mask = torch.empty(
            (2, 1, prefix_len, max_cache_len),
            dtype=mask_dtype,
            device=device,
        )
        self.prefill_pos = torch.zeros(2, prefix_len, dtype=torch.long, device=device)
        self.prefill_cache_position = torch.arange(prefix_len, device=device)
        self._refresh_prefix(cond_prefix, cond_audio, uncond_prefix, uncond_audio)

        # Fixed buffers for every denoising step. Values are refreshed in
        # begin_block(); shapes never change so the graph is captured once.
        self.step_ids = torch.full(
            (2, channels, block_size),
            model.config.audio_mask_id,
            dtype=torch.long,
            device=device,
        )
        self.step_audio = torch.ones(2, block_size, dtype=torch.bool, device=device)
        self.step_mask = torch.full(
            (2, 1, block_size, max_cache_len),
            neg,
            dtype=mask_dtype,
            device=device,
        )
        self.step_pos = torch.zeros(2, block_size, dtype=torch.long, device=device)
        self.cache_position = torch.zeros(block_size, dtype=torch.long, device=device)

        self._step_fn = self._eager_step
        if compile_steps:
            try:
                self._step_fn = torch.compile(
                    self._eager_step, mode="reduce-overhead", fullgraph=False
                )
            except Exception as exc:  # pragma: no cover - depends on torch build
                logger.warning("torch.compile setup failed (%s); using eager.", exc)
                self.compiled = False

    def _refresh_prefix(
        self, cond_prefix, cond_audio, uncond_prefix, uncond_audio
    ) -> None:
        """Reset the cache and prefill a new request into the static buffers."""
        lc = cond_prefix.shape[-1]
        lu = uncond_prefix.shape[-1]
        if lc > self.prefix_len or lu > self.prefix_len:
            raise ValueError("Request prefix exceeds the runner's allocated prefix.")
        self.lc = lc
        self.lu = lu
        self.cache.reset()
        self.ids.zero_()
        self.ids[0, :, :lc] = cond_prefix[0]
        self.ids[1, :, :lu] = uncond_prefix[0]
        self.audio.zero_()
        self.audio[0, :lc] = cond_audio[0]
        self.audio[1, :lu] = uncond_audio[0]
        self.prefill_mask.fill_(self.neg)
        self.prefill_mask[0, :, :, :lc] = 0
        self.prefill_mask[1, :, :, :lu] = 0
        if lu == 0:
            # Keep at least one key visible for the pad queries, otherwise SDPA
            # produces NaN rows that can leak into later attention.
            self.prefill_mask[1, :, :, 0] = 0
        self.prefill_pos.zero_()
        self.prefill_pos[0, :lc] = torch.arange(lc, device=self.ids.device)
        self.prefill_pos[1, :lu] = torch.arange(lu, device=self.ids.device)
        with torch.no_grad():
            self.model(
                input_ids=self.ids,
                audio_mask=self.audio,
                attention_mask=self.prefill_mask,
                position_ids=self.prefill_pos,
                past_key_values=self.cache,
                use_cache=True,
                cache_position=self.prefill_cache_position,
            )

    def reset(self, cond_prefix, cond_audio, uncond_prefix, uncond_audio) -> None:
        """Reuse this runner (and its captured graph) for another request."""
        self._refresh_prefix(cond_prefix, cond_audio, uncond_prefix, uncond_audio)

    def _eager_step(self):
        return self.model(
            input_ids=self.step_ids,
            audio_mask=self.step_audio,
            attention_mask=self.step_mask,
            position_ids=self.step_pos,
            past_key_values=self.cache,
            use_cache=True,
            cache_position=self.cache_position,
        ).logits

    def _run(self):
        if self.compiled:
            # Required by reduce-overhead: the previous replay's output tensor
            # is reused by the next replay.
            torch.compiler.cudagraph_mark_step_begin()
        return self._step_fn()

    def begin_block(self, block_index: int) -> None:
        end = self.prefix_len + block_index * self.block_size
        stop = end + self.block_size
        self.step_mask.fill_(self.neg)
        self.step_mask[0, :, :, : self.lc] = 0
        self.step_mask[1, :, :, : self.lu] = 0
        # Previously committed target blocks are stored contiguously at
        # [prefix_len, end); include them along with the active block.
        self.step_mask[0, :, :, self.prefix_len : stop] = 0
        self.step_mask[1, :, :, self.prefix_len : stop] = 0
        q = self.block_size
        self.step_pos[0] = (
            self.lc + block_index * q + torch.arange(q, device=self.step_pos.device)
        )
        self.step_pos[1] = (
            self.lu + block_index * q + torch.arange(q, device=self.step_pos.device)
        )
        self.cache_position[:] = torch.arange(
            end, stop, device=self.cache_position.device
        )

    def _fill_tokens(self, tokens) -> None:
        self.step_ids[0] = tokens[0]
        self.step_ids[1] = tokens[0]

    def step(self, tokens):
        self._fill_tokens(tokens)
        if self.compiled and not self._warmed:
            # Compile/capture once, before the timed part of the first block.
            try:
                self._run()
            except Exception as exc:  # pragma: no cover - depends on torch build
                logger.warning("torch.compile failed (%s); using eager.", exc)
                self.compiled = False
                self._step_fn = self._eager_step
            self._warmed = True
        logits = self._run()
        return logits[0:1], logits[1:2]

    def commit(self, tokens) -> None:
        self._fill_tokens(tokens)
        self._run()


def _get_compiled_runner(
    model,
    *,
    cond_prefix,
    cond_audio,
    uncond_prefix,
    uncond_audio,
    block_size: int,
    max_blocks: int,
    device,
):
    """Reuse one compiled runner per (device, shape, duration cap).

    ``torch.compile`` specializes on shapes, and creating a fresh runner per
    sentence would pay the compile/capture cost every time. The static buffers
    are request-independent, so a cached runner is reset and re-prefilled.
    """
    natural = max(cond_prefix.shape[-1], uncond_prefix.shape[-1])
    prefix_len = max(256, math.ceil(natural / 256) * 256)
    key = (
        model,
        str(device),
        block_size,
        max_blocks,
        prefix_len,
        str(next(model.parameters()).dtype),
    )
    runner = _COMPILED_RUNNERS.get(key)
    if runner is None:
        runner = _BatchedCFGRunner(
            model,
            cond_prefix=cond_prefix,
            cond_audio=cond_audio,
            uncond_prefix=uncond_prefix,
            uncond_audio=uncond_audio,
            block_size=block_size,
            max_blocks=max_blocks,
            compile_steps=True,
            device=device,
            prefix_len=prefix_len,
        )
        _COMPILED_RUNNERS[key] = runner
        # Keep memory bounded when a process sees many duration caps/shapes.
        while len(_COMPILED_RUNNERS) > 4:
            _COMPILED_RUNNERS.pop(next(iter(_COMPILED_RUNNERS)))
    else:
        runner.reset(cond_prefix, cond_audio, uncond_prefix, uncond_audio)
    return runner


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
    cond_audio = torch.zeros(1, cond_prefix.shape[-1], dtype=torch.bool, device=device)
    # Style and text occupy the non-audio prefix; prompt, if any, is the tail.
    prompt_len = ref_audio_tokens.shape[-1] if has_prompt else 0
    if prompt_len:
        cond_audio[0, -prompt_len:] = True

    use_cfg = gen_config.guidance_scale != 0
    if uncond_parts:
        uncond_prefix = torch.cat(uncond_parts, dim=2).to(device)
        uncond_audio = torch.ones(
            1, uncond_prefix.shape[-1], dtype=torch.bool, device=device
        )
    else:
        uncond_prefix = torch.empty(1, channels, 0, dtype=torch.long, device=device)
        uncond_audio = torch.empty(1, 0, dtype=torch.bool, device=device)

    max_frames = max(block_size, int(max_duration * frame_rate))
    max_blocks = math.ceil(max_frames / block_size)

    batch_cfg = bool(getattr(gen_config, "realtime_batch_cfg", True))
    compile_steps = bool(getattr(gen_config, "realtime_compile", False))
    if use_cfg and batch_cfg:
        if compile_steps:
            runner = _get_compiled_runner(
                model,
                cond_prefix=cond_prefix,
                cond_audio=cond_audio,
                uncond_prefix=uncond_prefix,
                uncond_audio=uncond_audio,
                block_size=block_size,
                max_blocks=max_blocks,
                device=device,
            )
        else:
            runner = _BatchedCFGRunner(
                model,
                cond_prefix=cond_prefix,
                cond_audio=cond_audio,
                uncond_prefix=uncond_prefix,
                uncond_audio=uncond_audio,
                block_size=block_size,
                max_blocks=max_blocks,
                compile_steps=False,
                device=device,
            )
    else:
        runner = _SeparateCFGRunner(
            model,
            cond_prefix=cond_prefix,
            cond_audio=cond_audio,
            uncond_prefix=uncond_prefix,
            uncond_audio=uncond_audio,
            use_cfg=use_cfg,
            block_size=block_size,
            device=device,
        )

    produced = 0
    layer_ids = torch.arange(channels, device=device).view(1, -1, 1)
    positions = block_size * channels
    counts = _unmask_counts(positions, gen_config.num_step, gen_config.t_shift, device)

    for block_index in range(max_blocks):
        tokens = torch.full(
            (1, channels, block_size),
            mask_id,
            dtype=torch.long,
            device=device,
        )
        runner.begin_block(block_index)
        remaining = positions
        for step, k in enumerate(counts):
            k = min(k, remaining)
            cond_logits, uncond_logits = runner.step(tokens)
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
                remaining=remaining,
            )
            remaining -= k
            if step == gen_config.num_step - 1:
                # The last step of the cosine schedule commits every remaining
                # mask. Force anything still masked (numerical ties) to the argmax.
                still = tokens == mask_id
                tokens = torch.where(still, pred, tokens)

        # Commit the clean block into the cache before reading the next one.
        runner.commit(tokens)

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
