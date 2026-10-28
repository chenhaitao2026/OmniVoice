#!/usr/bin/env python3
"""Evaluate the Luna-TTS Realtime streaming checkpoint on MiniMax's test set.

For each Chinese / English sentence in
https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set

1. clone the provided speaker prompt (zero-shot voice cloning),
2. synthesize with the block-causal streaming sampler
   (``OmniVoice.generate_realtime``),
3. transcribe the generated wav with the local OpenAI-compatible ASR service,
4. report WER (English, word level) / CER (Chinese, character level).

The ASR service is the one from the reference command::

    curl -X POST http://localhost:33601/v1/audio/transcriptions \
         -F file=@test.wav -F response_format=json

Usage:
    python examples/test_luna_realtime_minimax.py                # zh + en, 100x2
    python examples/test_luna_realtime_minimax.py --limit 10     # quick smoke test
    python examples/test_luna_realtime_minimax.py --langs zh     # Chinese only
    python examples/test_luna_realtime_minimax.py --no-stream    # decode at the end

The run is resumable: completed sample ids in ``results.jsonl`` are skipped,
so a failed or interrupted run can simply be started again. Each sample is
seeded from its id (``--seed``), so re-running reproduces the same audio.

Outputs (written under ``--output-dir``):
    wavs/<id>.wav       generated speech
    results.jsonl       per-sample reference / hypothesis / errors
    results.tsv         the same table, tab separated
    summary.json        per-language and macro-average error rates
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import unicodedata
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger("minimax_eval")

DEFAULT_DATA_ROOTS = [
    "/Cache/huggingface/hub/datasets--MiniMaxAI--TTS-Multilingual-Test-Set",
    str(
        Path.home()
        / ".cache/huggingface/hub/datasets--MiniMaxAI--TTS-Multilingual-Test-Set"
    ),
]

# CLI code -> (test-set directory / file stem, language id passed to the model)
LANGUAGES = {
    "zh": ("chinese", "zh"),
    "en": ("english", "en"),
}


@dataclass
class Sample:
    id: str
    lang: str
    speaker: str
    ref_audio: str
    ref_text: str
    text: str


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def resolve_dataset_root(root: Optional[str]) -> Path:
    """Accept either the HF cache repo dir or a snapshot dir."""
    candidates = [root] if root else DEFAULT_DATA_ROOTS
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if (path / "text").is_dir():
            return path
        snapshots = path / "snapshots"
        if snapshots.is_dir():
            for snap in sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime):
                if (snap / "text").is_dir():
                    return snap
    raise FileNotFoundError(
        "MiniMax TTS test set not found. Pass --dataset-root pointing to the "
        "HF repo dir (with snapshots/) or to a snapshot dir. Looked at: "
        + ", ".join(str(c) for c in candidates)
    )


def read_prompt_texts(path: Path) -> Dict[str, str]:
    """Map speaker audio file name -> reference transcript."""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "|" not in line:
                continue
            name, text = line.split("|", 1)
            mapping[name.strip()] = text.strip()
    return mapping


def find_speaker_audio(dataset: Path, lang_dir: str, speaker: str) -> Optional[Path]:
    speaker_dir = dataset / "speaker" / lang_dir / speaker
    if not speaker_dir.is_dir():
        return None
    for pattern in ("*.mp3", "*.wav", "*.flac", "*.m4a"):
        files = sorted(speaker_dir.glob(pattern))
        if files:
            return files[0]
    return None


def load_samples(
    dataset: Path,
    langs: Iterable[str],
    speakers: Iterable[str],
    limit: int = 0,
) -> List[Sample]:
    prompts = read_prompt_texts(dataset / "speaker" / "prompt_text.txt")
    samples: List[Sample] = []

    for cli_lang in langs:
        lang_dir, lang_id = LANGUAGES[cli_lang]
        text_file = dataset / "text" / f"{lang_dir}.txt"
        if not text_file.is_file():
            raise FileNotFoundError(f"Missing test text file: {text_file}")

        per_lang = 0
        with open(text_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line or "|" not in line:
                    continue
                speaker, text = line.split("|", 1)
                speaker = speaker.strip()
                text = text.strip()
                if speakers and speaker not in speakers:
                    continue
                ref_audio = find_speaker_audio(dataset, lang_dir, speaker)
                if ref_audio is None:
                    logger.warning("No prompt audio for speaker %s, skipping", speaker)
                    continue
                ref_text = prompts.get(ref_audio.name)
                if ref_text is None:
                    logger.warning("No prompt transcript for %s, skipping", ref_audio)
                    continue
                samples.append(
                    Sample(
                        id=f"{lang_id}_{idx:03d}_{speaker}",
                        lang=lang_id,
                        speaker=speaker,
                        ref_audio=str(ref_audio),
                        ref_text=ref_text,
                        text=text,
                    )
                )
                per_lang += 1
                if limit and per_lang >= limit:
                    break
        logger.info("Loaded %d %s samples", per_lang, cli_lang)
    return samples


# ---------------------------------------------------------------------------
# Text normalisation + edit distance
# ---------------------------------------------------------------------------
CN_DIGITS = "零一二三四五六七八九"
CN_UNITS = ["", "十", "百", "千"]


def int_to_chinese(number: int) -> str:
    """Format a non-negative integer the way Mandarin TTS normally reads it.

    Only used for the few test sentences containing Arabic digits (e.g.
    "65 公分"). Numbers above 9999 keep digit-by-digit form.
    """
    if number == 0:
        return "零"
    if number > 9999:
        return "".join(CN_DIGITS[int(ch)] for ch in str(number))

    digits = str(number)
    out = []
    pending_zero = False
    for i, ch in enumerate(digits):
        digit = int(ch)
        pos = len(digits) - 1 - i
        if digit == 0:
            pending_zero = True
            continue
        if pending_zero and out:
            out.append("零")
        pending_zero = False
        out.append(CN_DIGITS[digit] + CN_UNITS[pos])
    text = "".join(out)
    return text[1:] if text.startswith("一十") else text


def normalize_text(text: str, lang: str) -> str:
    """Lower-case, NFKC, drop punctuation/symbols, squeeze whitespace."""
    text = unicodedata.normalize("NFKC", text).lower()
    # (1998) style parentheticals that ASR usually omits
    text = re.sub(r"\([^)]*\d[^)]*\)", " ", text)
    if lang == "zh":
        # The ASR service writes numbers as Chinese numerals; the test set uses
        # Arabic digits. Convert the reference so both sides match.
        text = re.sub(r"\d+", lambda m: int_to_chinese(int(m.group())), text)
        text = text.replace("两", "二").replace("〇", "零")
    text = "".join(
        " " if unicodedata.category(ch)[0] in ("P", "S") else ch for ch in text
    )
    if lang == "zh":
        # Chinese is scored per character, so spaces carry no information.
        return re.sub(r"\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance_counts(ref: str, hyp: str, lang: str) -> Dict[str, int]:
    """Levenshtein distance with substitution/deletion/insertion breakdown."""
    ref_units = list(ref) if lang == "zh" else ref.split()
    hyp_units = list(hyp) if lang == "zh" else hyp.split()

    n, m = len(ref_units), len(hyp_units)
    # Each cell: (distance, substitutions, deletions, insertions)
    dp: List[List[Tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)
    ]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_units[i - 1] == hyp_units[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            sub = dp[i - 1][j - 1]
            dele = dp[i - 1][j]
            ins = dp[i][j - 1]
            best = min(
                (sub[0] + 1, 0, sub[1] + 1, sub[2], sub[3]),
                (dele[0] + 1, 1, dele[1], dele[2] + 1, dele[3]),
                (ins[0] + 1, 2, ins[1], ins[2], ins[3] + 1),
            )
            dp[i][j] = (best[0], best[2], best[3], best[4])

    _, subs, dels, ins = dp[n][m]
    return {
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "errors": subs + dels + ins,
        "ref_units": n,
    }


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------
def transcribe(
    wav_path: Path,
    url: str,
    timeout: float = 120.0,
    retries: int = 3,
) -> Tuple[str, Optional[str]]:
    """Call the local OpenAI-compatible transcription endpoint."""
    import requests

    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with open(wav_path, "rb") as f:
                resp = requests.post(
                    url,
                    files={"file": (wav_path.name, f, "audio/wav")},
                    data={"response_format": "json"},
                    timeout=timeout,
                )
            resp.raise_for_status()
            payload = resp.json()
            text = payload.get("text")
            if text is None:
                raise ValueError(f"ASR response has no 'text': {payload}")
            return text.strip(), payload.get("language")
        except Exception as exc:  # noqa: BLE001 - retried below
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"ASR failed for {wav_path}: {last_error}")


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
def configure_hf_cache(model_path: str) -> None:
    """Prefer the local HF cache when the checkpoint has no bundled tokenizer.

    Training checkpoints such as ``exp/luna_tts_realtime/checkpoint-20000`` do
    not store the Higgs audio tokenizer; ``OmniVoice.from_pretrained`` then
    calls ``snapshot_download("eustlb/higgs-audio-v2-tokenizer")``, which needs
    network access even when the files are cached.  If the tokenizer is already
    cached, switch huggingface_hub to offline mode *before* importing omnivoice
    so the lookup always succeeds on an air-gapped / flaky-network node.
    """
    if os.environ.get("HF_HUB_OFFLINE") is not None:
        return
    path = Path(model_path)
    if not path.is_dir() or (path / "audio_tokenizer").is_dir():
        return

    hub_dirs = [
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub",
        Path("/Cache/huggingface/hub"),
    ]
    for hub in hub_dirs:
        repo = hub / "models--eustlb--higgs-audio-v2-tokenizer" / "snapshots"
        if repo.is_dir() and any(repo.iterdir()):
            os.environ["HF_HUB_OFFLINE"] = "1"
            logger.info("Using cached audio tokenizer in offline mode (%s)", hub)
            return


def auto_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    best, best_free = "cuda:0", -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best, best_free = f"cuda:{i}", free
    return best


def synthesize(
    model: OmniVoice,
    sample: Sample,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, float, Optional[float]]:
    """Return (waveform, synthesis_seconds, time_to_first_chunk)."""
    gen_kwargs = dict(
        text=sample.text,
        language=sample.lang,
        ref_audio=sample.ref_audio,
        ref_text=sample.ref_text,
        guidance_scale=args.guidance_scale,
        t_shift=args.t_shift,
        denoise=args.denoise,
        postprocess_output=args.postprocess_output,
    )
    if args.num_step is not None:
        gen_kwargs["num_step"] = args.num_step

    start = time.perf_counter()
    ttfc: Optional[float] = None

    if args.stream:
        chunks: List[np.ndarray] = []
        # generate_realtime() returns a lazy generator; consume it inside
        # inference_mode so the generator body also runs without autograd.
        with torch.inference_mode():
            for chunk in model.generate_realtime(
                **gen_kwargs, max_duration=args.max_duration, stream=True
            ):
                if ttfc is None:
                    ttfc = time.perf_counter() - start
                chunks.append(np.asarray(chunk, dtype=np.float32))
        audio = (
            np.concatenate(chunks)
            if chunks
            else np.zeros(int(0.1 * model.sampling_rate), dtype=np.float32)
        )
    else:
        with torch.inference_mode():
            audio = model.generate_realtime(
                **gen_kwargs, max_duration=args.max_duration
            )[0]
        audio = np.asarray(audio, dtype=np.float32).squeeze()

    return audio, time.perf_counter() - start, ttfc


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def load_done_ids(path: Path) -> set:
    done = set()
    if not path.is_file():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    done.add(rec["id"])
            except Exception:  # noqa: BLE001 - tolerate a truncated last line
                continue
    return done


def summarize(records: List[dict]) -> dict:
    summary: Dict[str, dict] = {}
    for rec in records:
        if rec.get("status") != "ok":
            continue
        stats = summary.setdefault(
            rec["lang"],
            {
                "metric": "CER" if rec["lang"] == "zh" else "WER",
                "num_samples": 0,
                "errors": 0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "ref_units": 0,
                "audio_seconds": 0.0,
                "synth_seconds": 0.0,
                "ttfc_seconds": [],
            },
        )
        stats["num_samples"] += 1
        stats["errors"] += rec["errors"]
        stats["substitutions"] += rec["substitutions"]
        stats["deletions"] += rec["deletions"]
        stats["insertions"] += rec["insertions"]
        stats["ref_units"] += rec["ref_units"]
        stats["audio_seconds"] += rec["audio_duration"]
        stats["synth_seconds"] += rec["synth_time"]
        if rec.get("ttfc") is not None:
            stats["ttfc_seconds"].append(rec["ttfc"])

    for stats in summary.values():
        stats["error_rate"] = (
            100.0 * stats["errors"] / stats["ref_units"] if stats["ref_units"] else None
        )
        stats["rtf"] = (
            stats["synth_seconds"] / stats["audio_seconds"]
            if stats["audio_seconds"]
            else None
        )
        ttfc = stats.pop("ttfc_seconds")
        stats["mean_ttfc"] = float(np.mean(ttfc)) if ttfc else None

    rates = [s["error_rate"] for s in summary.values() if s["error_rate"] is not None]
    summary["macro_average"] = {
        "num_languages": len(rates),
        "error_rate": float(np.mean(rates)) if rates else None,
    }
    return summary


def print_summary(summary: dict) -> None:
    print("")
    print(
        f"{'Language':<10}{'N':>5}{'Metric':>8}{'Error%':>9}"
        f"{'Sub':>7}{'Del':>7}{'Ins':>7}{'Units':>8}{'RTF':>8}{'TTFC(s)':>9}"
    )
    print("-" * 87)
    for lang in ("zh", "en"):
        s = summary.get(lang)
        if not s or not s["num_samples"]:
            continue
        print(
            f"{lang:<10}{s['num_samples']:>5}{s['metric']:>8}"
            f"{s['error_rate']:>8.2f}%{s['substitutions']:>7}{s['deletions']:>7}"
            f"{s['insertions']:>7}{s['ref_units']:>8}"
            f"{(s['rtf'] or 0):>8.3f}"
            f"{(s['mean_ttfc'] if s['mean_ttfc'] is not None else float('nan')):>9.3f}"
        )
    macro = summary.get("macro_average", {})
    if macro.get("error_rate") is not None:
        print("-" * 87)
        print(
            f"{'Macro avg':<10}{'':>5}{'':>8}{macro['error_rate']:>8.2f}%"
            f"  ({macro['num_languages']} languages)"
        )
    print("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test Luna-TTS Realtime on MiniMax zh/en TTS test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent / "exp/luna_tts_realtime/checkpoint-20000"),
        help="Realtime checkpoint directory.",
    )
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("MINIMAX_TTS_DATASET_ROOT"),
        help="MiniMax TTS-Multilingual-Test-Set directory (snapshot or HF cache dir).",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["zh", "en"],
        choices=sorted(LANGUAGES),
        help="Languages to evaluate.",
    )
    parser.add_argument(
        "--speakers",
        nargs="*",
        default=[],
        help="Filter by speaker id, e.g. english_female chinese_male. Empty = all.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max sentences per language (0 = all)."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Default: visible GPU with the most free memory.",
    )
    parser.add_argument(
        "--asr-url",
        default="http://localhost:33601/v1/audio/transcriptions",
        help="OpenAI-compatible transcription endpoint.",
    )
    parser.add_argument("--asr-timeout", type=float, default=120.0)
    parser.add_argument("--asr-retries", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="results/luna_realtime_minimax",
        help="Where to write wavs and result tables.",
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        default=True,
        help="Use the incremental streaming API (chunks concatenated to a wav).",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Commit blocks internally but decode the whole utterance at the end.",
    )
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument(
        "--num-step",
        type=int,
        default=None,
        help="Refinement steps per block. Default: block_steps from checkpoint.",
    )
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--t-shift", type=float, default=0.1)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed. Each sample uses seed + CRC32(sample id) so a "
        "resumed run reproduces the same audio.",
    )
    parser.add_argument("--denoise", action="store_true", default=True)
    parser.add_argument("--no-denoise", dest="denoise", action="store_false")
    parser.add_argument(
        "--postprocess-output", action="store_true", default=True
    )
    parser.add_argument(
        "--no-postprocess-output", dest="postprocess_output", action="store_false"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-run samples already in results.jsonl."
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser


def main() -> int:
    args = get_parser().parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=getattr(logging, args.log_level),
        force=True,
    )

    dataset = resolve_dataset_root(args.dataset_root)
    logger.info("Dataset: %s", dataset)
    samples = load_samples(dataset, args.langs, args.speakers, args.limit)
    if not samples:
        logger.error("No samples selected.")
        return 1

    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = output_dir / "results.jsonl"
    if args.overwrite and results_jsonl.exists():
        results_jsonl.unlink()
    done_ids = load_done_ids(results_jsonl)

    pending = [s for s in samples if s.id not in done_ids]
    logger.info(
        "%d samples selected, %d already done, %d to run",
        len(samples),
        len(samples) - len(pending),
        len(pending),
    )

    model = None
    if pending:
        configure_hf_cache(args.model)
        # Imported here so configure_hf_cache() can set HF_HUB_OFFLINE first.
        from omnivoice import OmniVoice

        device = args.device or auto_device()
        logger.info("Loading %s on %s ...", args.model, device)
        model = OmniVoice.from_pretrained(
            args.model, device_map=device, dtype=torch.float16
        )
        logger.info(
            "Checkpoint: block_size=%s block_steps=%s audio_eos_id=%s",
            getattr(model.config, "block_size", None),
            getattr(model.config, "block_steps", None),
            getattr(model.config, "audio_eos_id", None),
        )

    sample_rate = model.sampling_rate if model is not None else 24000

    results_f = open(results_jsonl, "a", encoding="utf-8")
    try:
        for idx, sample in enumerate(pending, 1):
            wav_path = wav_dir / f"{sample.id}.wav"
            record = {
                "id": sample.id,
                "lang": sample.lang,
                "speaker": sample.speaker,
                "ref_audio": sample.ref_audio,
                "ref_text": sample.ref_text,
                "text": sample.text,
                "wav": str(wav_path),
            }
            try:
                torch.manual_seed(
                    (args.seed + zlib.crc32(sample.id.encode("utf-8"))) % (2**31 - 1)
                )
                audio, synth_time, ttfc = synthesize(model, sample, args)
                sf.write(wav_path, audio, sample_rate)
                audio_duration = len(audio) / sample_rate

                hypothesis, asr_language = transcribe(
                    wav_path,
                    args.asr_url,
                    timeout=args.asr_timeout,
                    retries=args.asr_retries,
                )
                norm_ref = normalize_text(sample.text, sample.lang)
                norm_hyp = normalize_text(hypothesis, sample.lang)
                counts = edit_distance_counts(norm_ref, norm_hyp, sample.lang)

                record.update(
                    status="ok",
                    hypothesis=hypothesis,
                    asr_language=asr_language,
                    norm_ref=norm_ref,
                    norm_hyp=norm_hyp,
                    audio_duration=audio_duration,
                    synth_time=synth_time,
                    ttfc=ttfc,
                    rtf=synth_time / audio_duration if audio_duration else None,
                    **counts,
                )
                logger.info(
                    "[%d/%d] %s  CER/WER=%5.2f%%  dur=%.2fs  rtf=%.3f%s",
                    idx,
                    len(pending),
                    sample.id,
                    100.0 * counts["errors"] / max(counts["ref_units"], 1),
                    audio_duration,
                    synth_time / audio_duration if audio_duration else float("nan"),
                    f"  ttfc={ttfc:.3f}s" if ttfc is not None else "",
                )
                if counts["errors"] and args.log_level == "DEBUG":
                    logger.debug("  ref: %s", norm_ref)
                    logger.debug("  hyp: %s", norm_hyp)
            except Exception as exc:  # noqa: BLE001 - keep the run going
                record.update(status="error", error=repr(exc))
                logger.exception("Sample %s failed", sample.id)

            results_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            results_f.flush()
    finally:
        results_f.close()

    # Re-read everything so resumed runs are included in the summary.  Keep the
    # last record per id so a retried sample does not double-count.
    records_by_id: Dict[str, dict] = {}
    with open(results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records_by_id[rec["id"]] = rec
    records = list(records_by_id.values())
    summary = summarize(records)
    total_failures = sum(1 for rec in records if rec.get("status") != "ok")

    tsv_path = output_dir / "results.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write(
            "id\tlang\tspeaker\tstatus\terror_rate\tsub\tde\tins\tunits\t"
            "audio_s\tsynth_s\tttfc_s\trtf\tref\thyp\n"
        )
        for rec in records:
            if rec.get("status") == "ok":
                rate = 100.0 * rec["errors"] / max(rec["ref_units"], 1)
                ttfc = "" if rec.get("ttfc") is None else f"{rec['ttfc']:.3f}"
                f.write(
                    f"{rec['id']}\t{rec['lang']}\t{rec['speaker']}\tok\t{rate:.2f}\t"
                    f"{rec['substitutions']}\t{rec['deletions']}\t{rec['insertions']}\t"
                    f"{rec['ref_units']}\t{rec['audio_duration']:.3f}\t"
                    f"{rec['synth_time']:.3f}\t{ttfc}\t"
                    f"{(rec.get('rtf') or 0):.3f}\t"
                    f"{rec['norm_ref']}\t{rec['norm_hyp']}\n"
                )
            else:
                f.write(
                    f"{rec['id']}\t{rec['lang']}\t{rec.get('speaker', '')}\terror"
                    + "\t" * 10
                    + f"{rec.get('error', '')}\n"
                )

    summary.update(
        model=args.model,
        dataset=str(dataset),
        streaming=args.stream,
        max_duration=args.max_duration,
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        t_shift=args.t_shift,
        seed=args.seed,
        num_selected=len(samples),
        num_results=len(records),
        num_failures=total_failures,
    )
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print_summary(summary)
    logger.info("Wrote %s, %s, %s", results_jsonl, tsv_path, output_dir / "summary.json")
    return 0 if total_failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
