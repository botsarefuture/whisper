import argparse
import os
import traceback
import warnings
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import torch
import tqdm

from .audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from .decoding import DecodingOptions, DecodingResult
from .timing import add_word_timestamps
from .tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from .utils import (
    exact_div,
    format_timestamp,
    get_end,
    get_writer,
    make_safe,
    optional_float,
    optional_int,
    str2bool,
)

if TYPE_CHECKING:
    from .model import Whisper


def transcribe(
    model: "Whisper",
    audio: Union[str, np.ndarray, torch.Tensor],
    *,
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
    word_timestamps: bool = False,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    clip_timestamps: Union[str, List[float]] = "0",
    hallucination_silence_threshold: Optional[float] = None,
    job_id: Optional[str] = None,
    status_path: Optional[str] = None,
    **decode_options,
):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model instance

    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successively used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    word_timestamps: bool
        Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
        and include the timestamps for each word in each segment.

    prepend_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the next word

    append_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the previous word

    initial_prompt: Optional[str]
        Optional text to provide as a prompt for the first window. This can be used to provide, or
        "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns
        to make it more likely to predict those word correctly.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    clip_timestamps: Union[str, List[float]]
        Comma-separated list start,end,start,end,... timestamps (in seconds) of clips to process.
        The last end timestamp defaults to the end of the file.

    hallucination_silence_threshold: Optional[float]
        When word_timestamps is True, skip silent periods longer than this threshold (in seconds)
        when a possible hallucination is detected

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """
    if status_path and job_id:
        with open(os.path.join(status_path, f"{job_id}.txt"), "w") as f:
            percentage_done = 0
            f.write(str(percentage_done)) # So the user won't be shown text: "In queue"
        
    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False

    # Pad 30-seconds of silence to the input audio, for slicing
    mel = log_mel_spectrogram(audio, model.dims.n_mels, padding=N_SAMPLES)
    content_frames = mel.shape[-1] - N_FRAMES
    content_duration = float(content_frames * HOP_LENGTH / SAMPLE_RATE)

    if decode_options.get("language", None) is None:
        if not model.is_multilingual:
            decode_options["language"] = "en"
        else:
            if verbose:
                print(
                    "Detecting language using up to the first 30 seconds. Use `--language` to specify the language"
                )
            mel_segment = pad_or_trim(mel, N_FRAMES).to(model.device).to(dtype)
            _, probs = model.detect_language(mel_segment)
            decode_options["language"] = max(probs, key=probs.get)
            if verbose is not None:
                print(
                    f"Detected language: {LANGUAGES[decode_options['language']].title()}"
                )
    
    language: str = decode_options["language"]
    task: str = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=language,
        task=task,
    )

    if isinstance(clip_timestamps, str):
        clip_timestamps = [
            float(ts) for ts in (clip_timestamps.split(",") if clip_timestamps else [])
        ]
    seek_points: List[int] = [round(ts * FRAMES_PER_SECOND) for ts in clip_timestamps]
    if len(seek_points) == 0:
        seek_points.append(0)
    if len(seek_points) % 2 == 1:
        seek_points.append(content_frames)
    seek_clips: List[Tuple[int, int]] = list(zip(seek_points[::2], seek_points[1::2]))

    punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"

    if word_timestamps and task == "translate":
        warnings.warn("Word-level timestamps on translations may not be reliable.")

    def decode_with_fallback(segment: torch.Tensor) -> DecodingResult:
        temperatures = (
            [temperature] if isinstance(temperature, (int, float)) else temperature
        )
        decode_result = None

        for t in temperatures:
            kwargs = {**decode_options}
            if t > 0:
                # disable beam_size and patience when t > 0
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
            else:
                # disable best_of when t == 0
                kwargs.pop("best_of", None)

            options = DecodingOptions(**kwargs, temperature=t)
            decode_result = model.decode(segment, options)

            needs_fallback = False
            if (
                compression_ratio_threshold is not None
                and decode_result.compression_ratio > compression_ratio_threshold
            ):
                needs_fallback = True  # too repetitive
            if (
                logprob_threshold is not None
                and decode_result.avg_logprob < logprob_threshold
            ):
                needs_fallback = True  # average log probability is too low
            if (
                no_speech_threshold is not None
                and decode_result.no_speech_prob > no_speech_threshold
            ):
                needs_fallback = False  # silence
            if not needs_fallback:
                break

        return decode_result

    clip_idx = 0
    seek = seek_clips[clip_idx][0]
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
        input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    if initial_prompt is not None:
        all_tokens.extend(tokenizer.encode(" " + initial_prompt.strip()))

    while clip_idx < len(seek_clips):
        seek_clip_start, seek_clip_end = seek_clips[clip_idx]
        if seek < seek_clip_start:
            seek = seek_clip_start
        if seek >= seek_clip_end:
            clip_idx += 1
            if clip_idx < len(seek_clips):
                seek = seek_clips[clip_idx][0]
            continue

        mel_segment = mel[:, seek:seek + N_FRAMES].to(model.device).to(dtype)
        result = decode_with_fallback(mel_segment)
        tokens = result.tokens
        segments = result.segments

        if word_timestamps:
            # handle word timestamps
            pass

        all_tokens.extend(tokens)
        all_segments.extend(segments)
        seek += N_FRAMES

        if status_path and job_id:
            percentage_done = (seek / content_frames) * 100
            with open(os.path.join(status_path, f"{job_id}.txt"), "w") as f:
                f.write(f"{percentage_done:.2f}")

    return {
        "text": tokenizer.decode(all_tokens),
        "segments": all_segments,
        "language": LANGUAGES[decode_options["language"]],
    }


def cli():
    parser = argparse.ArgumentParser(description="Transcribe audio files using Whisper.")
    parser.add_argument("audio", type=str, help="Path to the audio file to transcribe.")
    parser.add_argument(
        "--model", type=str, default="base", help="Model size (base, small, medium, large)."
    )
    parser.add_argument(
        "--language", type=str, help="Language of the audio. Leave empty to auto-detect."
    )
    parser.add_argument(
        "--task", type=str, choices=["transcribe", "translate"], default="transcribe",
        help="Task type: transcribe or translate."
    )
    parser.add_argument(
        "--temperature", type=float, nargs="+", default=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        help="Temperature(s) for decoding."
    )
    parser.add_argument(
        "--compression-ratio-threshold", type=float, default=2.4,
        help="Compression ratio threshold for fallback."
    )
    parser.add_argument(
        "--logprob-threshold", type=float, default=-1.0,
        help="Log probability threshold for fallback."
    )
    parser.add_argument(
        "--no-speech-threshold", type=float, default=0.6,
        help="No speech probability threshold for fallback."
    )
    parser.add_argument(
        "--condition-on-previous-text", type=str2bool, default=True,
        help="Whether to condition on previous text."
    )
    parser.add_argument(
        "--initial-prompt", type=str, help="Initial prompt for transcription."
    )
    parser.add_argument(
        "--word-timestamps", type=str2bool, default=False,
        help="Whether to include word-level timestamps."
    )
    parser.add_argument(
        "--prepend-punctuations", type=str, default="\"'“¿([{-",
        help="Punctuations to prepend."
    )
    parser.add_argument(
        "--append-punctuations", type=str, default="\"'.。,，!！?？:：”)]}、",
        help="Punctuations to append."
    )
    parser.add_argument(
        "--clip-timestamps", type=str, default="0",
        help="Comma-separated start,end timestamps for clips."
    )
    parser.add_argument(
        "--hallucination-silence-threshold", type=optional_float, default=None,
        help="Silence threshold for hallucination detection."
    )
    parser.add_argument(
        "--status-path", type=str, default=None,
        help="Path to the status file for reporting progress."
    )
    parser.add_argument(
        "--job-id", type=str, default=None,
        help="Job ID for progress reporting."
    )
    parser.add_argument(
        "--verbose", type=str2bool, default=None,
        help="Whether to display detailed output."
    )
    parser.add_argument(
        "--fp16", type=str2bool, default=True,
        help="Whether to use FP16 precision."
    )

    args = parser.parse_args()

    try:
        # Load model
        from whisper import Whisper  # Importing Whisper model
        model = Whisper.load(args.model)

        # Run transcription
        result = transcribe(
            model,
            args.audio,
            verbose=args.verbose,
            temperature=args.temperature,
            compression_ratio_threshold=args.compression_ratio_threshold,
            logprob_threshold=args.logprob_threshold,
            no_speech_threshold=args.no_speech_threshold,
            condition_on_previous_text=args.condition_on_previous_text,
            initial_prompt=args.initial_prompt,
            word_timestamps=args.word_timestamps,
            prepend_punctuations=args.prepend_punctuations,
            append_punctuations=args.append_punctuations,
            clip_timestamps=args.clip_timestamps,
            hallucination_silence_threshold=args.hallucination_silence_threshold,
            status_path=args.status_path,
            job_id=args.job_id,
            fp16=args.fp16,
            task=args.task,
        )
        # Output results
        print("Transcription result:")
        print("Text:", result["text"])
        print("Language:", result["language"])
        print("Segments:", result["segments"])

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    cli()
