#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Block-causal training samples for Luna-TTS Realtime.

Each utterance is split into fixed blocks of codec frames. An EOS frame is
always appended, then the tail is padded with EOS so the last block is full.
That keeps a terminal target even when speech already ends on a block boundary
(Luna-TTS Family, Sec. 3.2). Noise levels are sampled independently per block.
The model sees the BD3 two-stream concatenation

    xt = noisy target blocks
    x  = text/prompt context + clean target blocks

and is scored only on masked positions of xt, with ELBO weight 1/t.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

import torch

from omnivoice.models.block_diffusion import (
    CONTEXT_BLOCK_ID,
    MIN_MASK_T,
    PADDING_BLOCK_ID,
    STREAM_CLEAN,
    STREAM_NOISY,
    STREAM_PAD,
)


class BlockDiffusionSampleProcessor:
    """OmniVoice sample processor with per-block masking and a two-stream layout."""

    def __init__(
        self,
        text_tokenizer: Any,
        num_channels: int,
        audio_mask_id: int,
        audio_eos_id: int,
        block_size: int,
        prompt_ratio_range: tuple,
        mask_ratio_range: tuple,
        drop_cond_ratio: float,
        language_ratio: float,
        use_pinyin_ratio: float,
        instruct_ratio: float,
        only_instruct_ratio: float,
    ):
        del mask_ratio_range  # replaced by an independent t per block
        if block_size <= 0:
            raise ValueError("block_size must be positive for block-causal training.")
        if audio_eos_id == audio_mask_id:
            raise ValueError("audio_eos_id and audio_mask_id must differ.")
        self.text_tokenizer = text_tokenizer
        self.num_channels = num_channels
        self.audio_mask_id = audio_mask_id
        self.audio_eos_id = audio_eos_id
        self.block_size = block_size
        self.prompt_ratio_range = prompt_ratio_range
        self.drop_cond_ratio = drop_cond_ratio
        self.language_ratio = language_ratio
        self.use_pinyin_ratio = use_pinyin_ratio
        self.instruct_ratio = instruct_ratio
        self.only_instruct_ratio = only_instruct_ratio

    def _sample_t(self) -> float:
        return random.uniform(MIN_MASK_T, 1.0)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        drop_text = random.uniform(0, 1) < self.drop_cond_ratio
        if "clean_start_token_idx" in sample["label"]:
            drop_text = False
            prompt_length = int(sample["label"]["clean_start_token_idx"])
            use_language = random.uniform(0, 1) < self.language_ratio
            use_instruct = random.uniform(0, 1) < self.instruct_ratio
        elif drop_text:
            # CFG drops the text condition only. The acoustic prompt stays, so
            # the unconditional block geometry matches streaming inference.
            prompt_length = int(
                sample["audio_tokens"].shape[1]
                * random.uniform(*self.prompt_ratio_range)
            )
            use_language = False
            use_instruct = False
        else:
            prompt_ratio = random.uniform(*self.prompt_ratio_range)
            use_language = random.uniform(0, 1) < self.language_ratio
            use_instruct = random.uniform(0, 1) < self.instruct_ratio
            if use_instruct and random.uniform(0, 1) < self.only_instruct_ratio:
                prompt_ratio = 0.0
            prompt_length = int(sample["audio_tokens"].shape[1] * prompt_ratio)

        audio_tokens = sample["audio_tokens"].long()
        channels, total_frames = audio_tokens.shape
        prompt_length = min(max(prompt_length, 0), total_frames)
        prompt = audio_tokens[:, :prompt_length]
        speech = audio_tokens[:, prompt_length:]

        # Terminal EOS is present even when speech already fills a whole block.
        eos = torch.full((channels, 1), self.audio_eos_id, dtype=torch.long)
        target = torch.cat([speech, eos], dim=1)
        remainder = target.shape[1] % self.block_size
        if remainder:
            pad = self.block_size - remainder
            target = torch.cat(
                [
                    target,
                    torch.full((channels, pad), self.audio_eos_id, dtype=torch.long),
                ],
                dim=1,
            )

        n_blocks = target.shape[1] // self.block_size
        noisy = target.clone()
        xt_labels = torch.full_like(target, -100)
        # Per-frame ELBO weight. Unmasked codebooks are ignored via labels.
        xt_loss_weight = torch.zeros(target.shape[1], dtype=torch.float32)
        for block_idx in range(n_blocks):
            sl = slice(block_idx * self.block_size, (block_idx + 1) * self.block_size)
            t = self._sample_t()
            token_mask = torch.rand(channels, self.block_size) < t
            noisy[:, sl][token_mask] = self.audio_mask_id
            xt_labels[:, sl][token_mask] = target[:, sl][token_mask]
            xt_loss_weight[sl] = 1.0 / t

        style_ids, text_ids = self._condition_tokens(
            sample, drop_text, use_language, use_instruct
        )
        clean_audio = torch.cat([prompt, target], dim=1)
        if drop_text:
            clean = clean_audio
            clean_labels = torch.full_like(clean, -100)
            clean_audio_mask = torch.ones(clean.shape[1], dtype=torch.bool)
            clean_block_ids = torch.full(
                (clean.shape[1],), CONTEXT_BLOCK_ID, dtype=torch.long
            )
            if prompt_length:
                clean_block_ids[:prompt_length] = CONTEXT_BLOCK_ID
            clean_block_ids[prompt_length:] = torch.arange(n_blocks).repeat_interleave(
                self.block_size
            )
            clean_pos = torch.arange(clean.shape[1], dtype=torch.long)
            target_origin = prompt_length
        else:
            clean = torch.cat([style_ids, text_ids, clean_audio], dim=1)
            clean_labels = torch.full_like(clean, -100)
            text_len = style_ids.shape[1] + text_ids.shape[1]
            clean_audio_mask = torch.zeros(clean.shape[1], dtype=torch.bool)
            clean_audio_mask[text_len:] = True
            clean_block_ids = torch.full(
                (clean.shape[1],), CONTEXT_BLOCK_ID, dtype=torch.long
            )
            clean_block_ids[text_len + prompt_length :] = torch.arange(
                n_blocks
            ).repeat_interleave(self.block_size)
            clean_pos = torch.arange(clean.shape[1], dtype=torch.long)
            target_origin = text_len + prompt_length

        # Noisy-stream positions match the positions those frames occupy in the
        # clean stream, which is also where inference places the active block.
        xt_pos = torch.arange(target.shape[1], dtype=torch.long) + target_origin
        xt_block_ids = torch.arange(n_blocks).repeat_interleave(self.block_size)
        xt_audio_mask = torch.ones(target.shape[1], dtype=torch.bool)

        input_ids = torch.cat([noisy, clean], dim=1)
        labels = torch.cat([xt_labels, clean_labels], dim=1)
        audio_mask = torch.cat([xt_audio_mask, clean_audio_mask], dim=0)
        block_ids = torch.cat([xt_block_ids, clean_block_ids], dim=0)
        stream_ids = torch.cat(
            [
                torch.full((target.shape[1],), STREAM_NOISY, dtype=torch.long),
                torch.full((clean.shape[1],), STREAM_CLEAN, dtype=torch.long),
            ]
        )
        position_ids = torch.cat([xt_pos, clean_pos], dim=0)
        loss_weight = torch.cat(
            [xt_loss_weight, torch.zeros(clean.shape[1], dtype=torch.float32)]
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "audio_mask": audio_mask,
            "block_ids": block_ids,
            "stream_ids": stream_ids,
            "position_ids": position_ids,
            "loss_weight": loss_weight,
            "length": input_ids.shape[1],
        }

    def _condition_tokens(self, sample, drop_text, use_language, use_instruct):
        if drop_text:
            empty = torch.empty(self.num_channels, 0, dtype=torch.long)
            return empty, empty

        language = (
            sample["label"].get("language_id", "None") if use_language else "None"
        )
        instruct = sample["label"].get("instruct", "None") if use_instruct else "None"
        style = f"<|lang_start|>{language}<|lang_end|>"
        style += f"<|instruct_start|>{instruct}<|instruct_end|>"
        if "clean_start_token_idx" in sample["label"]:
            style = "<|denoise|>" + style
        style_ids = self.text_tokenizer(style, return_tensors="pt").input_ids.repeat(
            self.num_channels, 1
        )
        if (
            "text_pinyin" in sample["label"]
            and random.uniform(0, 1) < self.use_pinyin_ratio
        ):
            text = sample["label"]["text_pinyin"]
        else:
            text = sample["label"]["text"]
        text_ids = self.text_tokenizer(
            f"<|text_start|>{text}<|text_end|>", return_tensors="pt"
        ).input_ids.repeat(self.num_channels, 1)
        return style_ids, text_ids


class BlockPaddingDataCollator:
    """Pad block-causal samples to ``[B, ...]`` and isolate them by document id."""

    def __init__(self, processor, batch_tokens: int):
        self.processor = processor
        self.batch_tokens = batch_tokens

    def __call__(self, processed_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        pad_id = self.processor.text_tokenizer.pad_token_id
        max_len = max(s["length"] for s in processed_samples)
        batch = len(processed_samples)

        def pad_channels(key, value):
            rows = []
            for sample in processed_samples:
                pad = max_len - sample["length"]
                rows.append(
                    torch.nn.functional.pad(sample[key], (0, pad), value=value)
                )
            return torch.stack(rows, dim=0)

        input_ids = pad_channels("input_ids", pad_id)
        labels = pad_channels("labels", -100)
        audio_mask = pad_channels("audio_mask", False)
        position_ids = pad_channels("position_ids", 0)
        block_ids = pad_channels("block_ids", PADDING_BLOCK_ID)
        stream_ids = pad_channels("stream_ids", STREAM_PAD)
        loss_weight = pad_channels("loss_weight", 0.0)

        document_ids = torch.zeros(batch, max_len, dtype=torch.long)
        for i, sample in enumerate(processed_samples):
            document_ids[i, : sample["length"]] = i
            document_ids[i, sample["length"] :] = -1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "audio_mask": audio_mask,
            "position_ids": position_ids,
            "block_ids": block_ids,
            "stream_ids": stream_ids,
            "loss_weight": loss_weight,
            "document_ids": document_ids,
        }


class BlockPackingDataCollator:
    """Pack block-causal samples into one sequence for flex attention."""

    def __init__(self, processor, batch_tokens: int):
        self.processor = processor
        self.batch_tokens = batch_tokens

    def __call__(self, processed_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        target = self.batch_tokens
        input_ids = torch.cat([s["input_ids"] for s in processed_samples], dim=1)
        labels = torch.cat([s["labels"] for s in processed_samples], dim=1)
        audio_mask = torch.cat([s["audio_mask"] for s in processed_samples], dim=0)
        position_ids = torch.cat([s["position_ids"] for s in processed_samples], dim=0)
        block_ids = torch.cat([s["block_ids"] for s in processed_samples], dim=0)
        stream_ids = torch.cat([s["stream_ids"] for s in processed_samples], dim=0)
        loss_weight = torch.cat([s["loss_weight"] for s in processed_samples], dim=0)

        document_ids = torch.cat(
            [
                torch.full((s["length"],), i, dtype=torch.long)
                for i, s in enumerate(processed_samples)
            ],
            dim=0,
        )
        pad = target - input_ids.shape[1]
        pad_id = self.processor.text_tokenizer.pad_token_id
        input_ids = torch.nn.functional.pad(input_ids, (0, pad), value=pad_id)
        labels = torch.nn.functional.pad(labels, (0, pad), value=-100)
        audio_mask = torch.nn.functional.pad(audio_mask, (0, pad), value=False)
        position_ids = torch.nn.functional.pad(position_ids, (0, pad), value=0)
        block_ids = torch.nn.functional.pad(
            block_ids, (0, pad), value=PADDING_BLOCK_ID
        )
        stream_ids = torch.nn.functional.pad(stream_ids, (0, pad), value=STREAM_PAD)
        loss_weight = torch.nn.functional.pad(loss_weight, (0, pad), value=0.0)
        document_ids = torch.nn.functional.pad(document_ids, (0, pad), value=-1)

        return {
            "input_ids": input_ids.unsqueeze(0),
            "labels": labels.unsqueeze(0),
            "audio_mask": audio_mask.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
            "block_ids": block_ids.unsqueeze(0),
            "stream_ids": stream_ids.unsqueeze(0),
            "loss_weight": loss_weight.unsqueeze(0),
            "document_ids": document_ids.unsqueeze(0),
        }
