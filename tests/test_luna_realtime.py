#!/usr/bin/env python3
"""Geometry and smoke tests for Luna-TTS Realtime block diffusion.

Runs without pytest: ``python tests/test_luna_realtime.py``.
"""

import unittest

import torch
from transformers import AutoModel, Qwen3Config

from omnivoice.data.block_processor import (
    BlockDiffusionSampleProcessor,
    BlockPaddingDataCollator,
)
from omnivoice.models.block_diffusion import (
    CONTEXT_BLOCK_ID,
    STREAM_CLEAN,
    STREAM_NOISY,
    build_block_attention_mask,
    iter_realtime_blocks,
    resize_audio_vocab,
)
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig, OmniVoiceGenerationConfig


class _FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [1, 2, 3][: max(1, min(3, len(text)))]
        if return_tensors == "pt":
            return type("Batch", (), {"input_ids": torch.tensor([ids], dtype=torch.long)})()
        return type("Batch", (), {"input_ids": ids})()


def _tiny_model(vocab=16, mask_id=14, codebooks=2):
    llm_config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        use_sliding_window=False,
    )
    llm = AutoModel.from_config(llm_config, attn_implementation="sdpa")
    model = OmniVoice(
        OmniVoiceConfig(
            audio_vocab_size=vocab,
            audio_mask_id=mask_id,
            num_audio_codebook=codebooks,
            audio_codebook_weights=[1.0] * codebooks,
            llm_config=llm.config,
        ),
        llm=llm,
    )
    model.llm.config._attn_implementation = "sdpa"
    model.eval()
    return model


class BlockMaskTest(unittest.TestCase):
    def test_two_stream_rules_and_document_isolation(self):
        # One packed row: noisy b0, noisy b1, context, clean b0, clean b1, then
        # a second document that must be invisible.
        block_ids = torch.tensor([[0, 1, CONTEXT_BLOCK_ID, 0, 1, 0, CONTEXT_BLOCK_ID]])
        stream_ids = torch.tensor(
            [[STREAM_NOISY, STREAM_NOISY, STREAM_CLEAN, STREAM_CLEAN, STREAM_CLEAN, STREAM_NOISY, STREAM_CLEAN]]
        )
        document_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1]])
        mask = build_block_attention_mask(block_ids, stream_ids, document_ids)[0, 0]

        def attends(q, keys):
            return [bool(mask[q, k]) for k in keys]

        self.assertEqual(attends(0, [0, 1, 2, 3, 4, 5, 6]), [True, False, True, False, False, False, False])
        self.assertEqual(attends(1, [0, 1, 2, 3, 4]), [False, True, True, True, False])
        self.assertEqual(attends(2, [2, 3, 4]), [True, False, False])
        self.assertEqual(attends(3, [2, 3, 4]), [True, True, False])
        self.assertEqual(attends(4, [2, 3, 4]), [True, True, True])
        self.assertFalse(bool(mask[5, 0]))
        self.assertFalse(bool(mask[0, 6]))


class ProcessorTest(unittest.TestCase):
    def _processor(self, block_size=4):
        proc = BlockDiffusionSampleProcessor(
            text_tokenizer=_FakeTokenizer(),
            num_channels=2,
            audio_mask_id=14,
            audio_eos_id=15,
            block_size=block_size,
            prompt_ratio_range=(0.0, 0.0),
            mask_ratio_range=(0.0, 1.0),
            drop_cond_ratio=0.0,
            language_ratio=0.0,
            use_pinyin_ratio=0.0,
            instruct_ratio=0.0,
            only_instruct_ratio=0.0,
        )
        proc._sample_t = lambda: 0.5
        return proc

    def test_eos_padding_and_position_alignment(self):
        speech = torch.arange(8, dtype=torch.long).view(2, 4)  # exactly one block
        sample = {
            "audio_tokens": speech,
            "label": {"text": "hello", "language_id": "en"},
        }
        out = self._processor().__call__(sample)
        block_size = 4
        # +1 EOS frame, padded out to a second block.
        n_target = 8
        xt = out["stream_ids"] == STREAM_NOISY
        self.assertEqual(int(xt.sum()), n_target)
        clean_target = (out["stream_ids"] == STREAM_CLEAN) & (out["block_ids"] >= 0)
        self.assertEqual(int(clean_target.sum()), n_target)
        self.assertTrue(torch.equal(out["position_ids"][xt], out["position_ids"][clean_target]))
        # Second block is pure EOS.
        eos_block = out["input_ids"][:, xt][:, block_size:]
        # Noisy stream may be masked; the clean copy is the supervision.
        clean_eos = out["input_ids"][:, clean_target][:, block_size:]
        self.assertTrue(torch.equal(clean_eos, torch.full_like(clean_eos, 15)))
        self.assertTrue((out["labels"][:, out["stream_ids"] == STREAM_CLEAN] == -100).all())
        masked = out["labels"][:, xt] != -100
        self.assertTrue(masked.any())
        self.assertTrue((out["loss_weight"][xt] == 2.0).all())
        # Masked noisy labels recover the clean token.
        self.assertTrue(
            torch.equal(
                out["labels"][:, xt][masked],
                out["input_ids"][:, clean_target][masked],
            )
        )
        del eos_block

    def test_collator_runs_forward(self):
        proc = self._processor()
        sample = {
            "audio_tokens": torch.randint(0, 10, (2, 5)),
            "label": {"text": "hi"},
        }
        batch = BlockPaddingDataCollator(proc, batch_tokens=256)(
            [proc(sample), proc(sample)]
        )
        model = _tiny_model()
        out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        self.assertEqual(out.logits.shape[1], 2)


class RealtimeSmokeTest(unittest.TestCase):
    def test_resize_keeps_old_codebook_rows(self):
        model = _tiny_model(vocab=8, mask_id=6, codebooks=2)
        with torch.no_grad():
            model.audio_embeddings.weight.copy_(torch.arange(model.audio_embeddings.weight.shape[0]).unsqueeze(1))
        resize_audio_vocab(model, 9)
        # layer 1, token 0 was at index 8 and must move to index 9.
        self.assertEqual(int(model.audio_embeddings.weight[9, 0]), 8)
        self.assertEqual(int(model.codebook_layer_offsets[1]), 9)
        self.assertEqual(model.config.audio_vocab_size, 9)

    def test_cfg_does_not_collapse_onto_eos(self):
        model = _tiny_model(vocab=16, mask_id=14, codebooks=2)
        model.config.audio_eos_id = 15
        # Conditional model prefers speech token 3, with a moderate EOS logit.
        # Unconditional model also prefers 3, but assigns EOS a large negative
        # logit — the gap that used to make guidance emit EOS immediately.
        cond = torch.full((1, 2, 4, 16), -2.0)
        uncond = torch.full((1, 2, 4, 16), -2.0)
        cond[..., 3] = 4.0
        cond[..., 15] = 1.0
        uncond[..., 3] = 4.0
        uncond[..., 15] = -8.0
        gen = OmniVoiceGenerationConfig(guidance_scale=2.0, class_temperature=0.0)
        pred, _ = model._predict_tokens_with_scoring(cond, uncond, gen)
        self.assertTrue(torch.equal(pred, torch.full_like(pred, 3)))

        # A conditional model that is actually done should still be allowed to stop.
        cond[..., 15] = 8.0
        cond[..., 3] = 0.0
        pred, _ = model._predict_tokens_with_scoring(cond, uncond, gen)
        self.assertTrue(torch.equal(pred, torch.full_like(pred, 15)))

    def test_one_block_with_and_without_cfg(self):
        model = _tiny_model(vocab=16, mask_id=14, codebooks=2)
        model.config.block_size = 4
        model.config.audio_eos_id = 15
        model.config.block_steps = 1
        model.text_tokenizer = _FakeTokenizer()
        prompt = torch.randint(0, 10, (2, 3))
        for scale in (0.0, 2.0):
            gen = OmniVoiceGenerationConfig(num_step=1, guidance_scale=scale, position_temperature=0.0)
            blocks = list(
                iter_realtime_blocks(
                    model,
                    text="hello",
                    ref_text=None,
                    ref_audio_tokens=prompt,
                    lang="en",
                    instruct=None,
                    gen_config=gen,
                    max_duration=0.2,
                    frame_rate=25.0,
                )
            )
            if blocks:
                self.assertEqual(blocks[0].shape[0], 2)
                self.assertLessEqual(blocks[0].shape[1], 4)
                self.assertFalse(torch.isin(blocks[0], torch.tensor(15)).any())


if __name__ == "__main__":
    unittest.main()
