# Luna-TTS Realtime

Luna-TTS Realtime is the block-causal streaming variant of the Luna-TTS family
([arXiv:2608.11593](https://arxiv.org/abs/2608.11593)). OmniVoice is already the
fully bidirectional masked-diffusion operating point (Luna-TTS / OmniVoice).
Realtime is obtained from that checkpoint by continual training, not by a
separate architecture.

## What changes

| | OmniVoice | Luna-TTS Realtime |
|---|---|---|
| Attention | bidirectional over the whole utterance | block-causal: bidirectional inside a block, causal across blocks |
| Block | whole utterance | 32 codec frames (1.28 s at 25 Hz) |
| Length | duration estimator | learned audio EOS, with `max_duration` as a cap |
| Cache | none | KV cache of committed blocks |
| Training noise | one mask rate per utterance | an independent mask rate per block |
| Objective | masked cross-entropy | same loss, weighted by `1/t` on the noisy stream only |

The speech tokenizer, text tokenizer, codebook embeddings, and parallel codebook
heads are reused. Training adds one audio EOS id (`1025` by default) and a
BD3-LM two-stream sequence:

```
[noisy target blocks | text + prompt + clean target blocks]
```

Noisy block `b` attends to itself and to clean tokens from blocks `< b`.
Clean tokens attend only to clean tokens up to their own block. An EOS frame is
appended even when speech already ends on a block boundary, then the tail is
padded with EOS so every block is full.

Inference prefills text and the reference prompt once, denoises the active
block for `block_steps` (default 8) confidence-unmasking steps, commits the
clean block into the KV cache, and repeats until EOS. Classifier-free guidance
drops the text condition and keeps the acoustic prompt, matching training.

This is not a port of Luna-Codec, the token-level duration predictor, or the
paper's GRPO stage. Those are separate from the realtime decoding regime.

## Train

Start from a bidirectional OmniVoice checkpoint. `examples/config/train_config_luna_realtime.json`
sets `block_causal: true`, `block_size: 32`, and about 20K adaptation steps,
which is the block-adaptation budget in the paper. Data prep is unchanged:
tokenize with `omnivoice.scripts.extract_audio_tokens`, then

```bash
cd examples
GPU_IDS=0,1 NUM_GPUS=2 ./run_luna_realtime.sh
```

`attn_implementation` may be `"sdpa"` or `"flex_attention"`. Both build the
same block-causal mask. SDPA is the safer default.

The saved config records `block_size`, `audio_eos_id`, and `block_steps`. A
normal OmniVoice checkpoint does not have these and cannot stream.

## Infer

```bash
omnivoice-infer --model exp/luna_tts_realtime/checkpoint-20000 \
    --text "你好，这是流式合成。" \
    --ref_audio ref.wav --ref_text "参考音频文本。" \
    --realtime --num_step 8 --output out.wav
```

`--num_step` is the number of refinement steps inside each block, not the
number of blocks. Omit it to use `block_steps` from the checkpoint (8).
`--max_duration` (or `--duration`) only caps the utterance; the model stops
itself on EOS. If the checkpoint predicts EOS at the very start — producing a
near-silent fragment — `--min_frames` sets a floor: EOS logits are masked out
until that many codec frames have been committed (25 frames = 1 s). The same
option is available as `min_frames` in `generate_realtime()` and in
`OmniVoiceGenerationConfig`; the default is 5 frames, and 0 disables the floor.

```python
from omnivoice import OmniVoice

model = OmniVoice.from_pretrained("exp/luna_tts_realtime/checkpoint-20000", device_map="cuda:0")
audio = model.generate_realtime(
    text="Hello, this is block-causal streaming TTS.",
    ref_audio="ref.wav",
    ref_text="Reference transcript.",
    num_step=8,
    max_duration=30,
    min_frames=25,
)
# Incremental delivery. Each chunk is the new tail of a cumulative codec decode,
# so concatenation equals a single full decode.
for chunk in model.generate_realtime(text="...", ref_audio="ref.wav", ref_text="...", stream=True):
    play(chunk)
```

`generate()` is unchanged and still runs the original bidirectional sampler.
Do not point it at a realtime checkpoint if you want the streaming schedule;
the weights can still run that path, but they were adapted for block-causal
attention.

## Inference performance

Three decoding paths are available:

* **Separate CFG** (`realtime_batch_cfg=False`): the original sampler, one
  forward per condition stream.
* **Batched CFG** (default): `cond` and `uncond` run as one `batch=2` forward on
  a preallocated `StaticCache`. A 32-frame step is kernel-launch bound, so
  batch=2 costs essentially the same as batch=1 and roughly halves the wall
  time.
* **Batched CFG + CUDA graph** (`realtime_compile=True`, or
  `--realtime_compile`): each denoising step (including the KV-cache writes) is
  captured with `torch.compile(mode="reduce-overhead")`. Replay removes the
  ~1.7k tiny kernel launches that leave the GPU idle in eager mode. The first
  request pays a one-off compile (tens of seconds on a cold Inductor cache);
  later requests reuse the graph. Prefix lengths are bucketed to 256 tokens so
  a serving process builds only a handful of graphs.

```python
for chunk in model.generate_realtime(
    text="...", ref_audio="ref.wav", ref_text="...",
    stream=True, realtime_compile=True,
):
    play(chunk)
```

Measured on one A100, fp16, single stream, `max_duration=30`, 8 steps per
block, 30 Chinese + 30 English sentences, including reference-audio encoding,
prefill, codec decode and streaming cross-fade. The compile time of the first
request is excluded:

| mode | RTF | TTFC |
|---|---|---|
| separate CFG (old) | 0.55–0.57 | 0.81 s |
| batched CFG | 0.28–0.30 | 0.43–0.48 s |
| batched CFG + CUDA graph | 0.07–0.08 | 0.17–0.19 s |

CFG batching changes the fp16 reduction order, so output is not bit-identical
to the unbatched sampler; CER/WER is unchanged within evaluation noise. After
the incremental decode below, the remaining costs are the one-off prefill
(TTFC) and the per-block Transformer step.

### Incremental codec decode

The DAC codec decoder is local: re-decoding a fixed window of
`stream_decode_context` seconds (default 2.56 s) with each new block produces
the same waveform as re-decoding the whole utterance, at ~58–61 dB SNR, while
turning streaming decode from O(T²) into O(T). On a 20 s utterance the
end-to-end RTF drops from 0.077 (cumulative decode) to 0.061. Set
`stream_decode_context=0` to restore the old cumulative decode.

The remaining single-stream cost is the per-block Transformer step: roughly
5 ms per denoising step on an A100, i.e. ~0.03 RTF for the denoiser alone.
Getting to the low-0.02 range on one stream needs either a more efficient
long-context attention (the dense block mask makes the SDPA kernel inefficient
at 32 queries) or batching several streams; on H100 / batch≥2 the same path is
well below 0.024.
