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
itself on EOS.

```python
from omnivoice import OmniVoice

model = OmniVoice.from_pretrained("exp/luna_tts_realtime/checkpoint-20000", device_map="cuda:0")
audio = model.generate_realtime(
    text="Hello, this is block-causal streaming TTS.",
    ref_audio="ref.wav",
    ref_text="Reference transcript.",
    num_step=8,
    max_duration=30,
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
