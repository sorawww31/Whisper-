import argparse
import logging
import os
import time
from datetime import timedelta

import torch
import whisper
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from moviepy import VideoFileClip
from pyannote.audio import Audio, Pipeline
from pyannote.core import Segment
from tqdm import tqdm

logging.getLogger().setLevel(logging.WARNING)


def convert_mp4_to_wav(input_file="./ookawa.mp4", output_file="./ookawa_edited.wav"):
    """
    Convert only the first segment (e.g., 5 minutes) of an MP4 video file to a WAV audio file.

    Parameters:
    input_file (str): Path to the input MP4 file.
    output_file (str): Path where the output WAV file will be saved.
    segment_duration (int): Duration (in seconds) of the segment to extract. Default is 300 (5 minutes).
    """
    print("Converting MP4 to WAV (first 5 minutes)...")
    start_time = time.perf_counter()

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    if os.path.exists(output_file):
        return

    video_clip = VideoFileClip(input_file)
    total_duration = video_clip.duration

    segment_clip = video_clip.subclipped(0, total_duration)
    segment_clip.audio.write_audiofile(
        output_file, codec="pcm_s16le", ffmpeg_params=["-ac", "1", "-ar", "16000"]
    )
    segment_clip.close()
    video_clip.close()

    elapsed = time.perf_counter() - start_time
    print(f"MP4 to WAV conversion completed in {elapsed:.2f} seconds.\n")


def model_load(args, model_size="large-v2", device="cuda", compute_type="int8_float16"):
    """
    Load the Whisper model and pyannote pipeline.

    Returns:
    (model, pipeline): Loaded Whisper model and pyannote Pipeline.
    """
    print("Loading Whisper model and pyannote pipeline...")
    start_time = time.perf_counter()

    # Load .env and retrieve token
    load_dotenv()
    hf_token = os.getenv("HF_HUB_TOKEN")
    if not hf_token:
        raise ValueError(
            "No HF_HUB_TOKEN found in environment variables. Please set it before running the script."
        )
    os.environ["HF_HUB_TOKEN"] = hf_token

    # Load the faster-Whisper model or standard whisper based on args
    if args.faster:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root="whisper_models",
        )
    else:
        model = whisper.load_model(
            args.model_size, device=args.device, download_root="whisper_models"
        )

    # Load pyannote's speaker diarization pipeline (with proper authentication)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    if device == "cuda":
        # Move the pipeline to GPU if available
        pipeline.to(torch.device("cuda"))

    elapsed = time.perf_counter() - start_time
    print(f"Model and pipeline loaded in {elapsed:.2f} seconds.\n")
    return model, pipeline


def segments_transform(segments):
    segments_transformed = []
    segments_list = sorted(list(segments), key=lambda x: x[0].end)
    segment_0, _, speaker_0 = segments_list[0]
    previous_speaker = speaker_0
    start = segment_0.start
    end = segment_0.end
    for segment, _, speaker in segments_list:
        if speaker == previous_speaker:
            end = segment.end
        else:
            new_segment = Segment(start, end)
            if new_segment.duration >= 0.2:
                segments_transformed.append((new_segment, previous_speaker))
            previous_speaker = speaker
            start = segment.start
            end = segment.end

    return segments_transformed


def format_time(seconds):
    # 秒数を整数に変換
    total_seconds = int(seconds)
    # timedeltaオブジェクトを作成
    td = timedelta(seconds=total_seconds)
    # 時間、分、秒を取得
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    # フォーマットされた文字列を返す
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def main(args):
    total_start = time.perf_counter()
    input_file = args.input_file
    wav_file = args.wav_file

    # Convert MP4 to WAV and time it
    convert_mp4_to_wav(input_file, wav_file)

    # Load models and measure time
    model, pipeline = model_load(
        args,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )

    print("Running speaker diarization...")
    diarization_start = time.perf_counter()
    if args.num_speakers:
        diarization = pipeline(wav_file, num_speakers=args.num_speakers)
    else:
        diarization = pipeline(wav_file)

    diarization_elapsed = time.perf_counter() - diarization_start
    print(f"Speaker diarization completed in {diarization_elapsed:.2f} seconds.\n")

    audio = Audio(sample_rate=16000, mono=True)

    segments = diarization.itertracks(yield_label=True)
    segments = segments_transform(segments)
    merged_segments = []
    whisper_start = time.perf_counter()
    for segment, speaker in tqdm(segments, desc="Transcribing segments"):
        waveform, sample_rate = audio.crop(wav_file, segment)

        # model.transcribe returns a tuple: (segments, info)
        if args.faster:
            segments_transcription, info = model.transcribe(
                waveform.squeeze().numpy(), language=args.language
            )
        else:
            segments_transcription = model.transcribe(
                waveform.squeeze().numpy(), language=args.language
            )
        # 複数のセグメントが返る場合、各テキストを結合（必要に応じて処理を調整）
        if args.faster:
            text = " ".join([seg.text for seg in segments_transcription])
        else:
            text = segments_transcription["text"]

        seg = {
            "start": segment.start,
            "end": segment.end,
            "speaker": speaker,
            "text": text,
        }
        merged_segments.append(seg)

    total_elapsed = time.perf_counter() - total_start
    whisper_elapsed = time.perf_counter() - whisper_start
    print(f"Transcription completed in {whisper_elapsed:.2f} seconds.\n")
    print(f"\nTotal processing time: {total_elapsed:.2f} seconds.")

    output_txt = args.output_file
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"Input file: {input_file}\n")
        f.write(f"Output file: {wav_file}\n")
        f.write(f"Model size: {args.model_size}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Compute type: {args.compute_type}\n")
        f.write(f"Language: {args.language}\n")
        f.write(
            f"Speaker diarization completed in {diarization_elapsed:.2f} seconds.\n"
        )
        f.write(f"Transcription completed in {whisper_elapsed:.2f} seconds.\n")
        f.write(f"\nTotal processing time: {total_elapsed:.2f} seconds.\n\n")

        for seg in merged_segments:
            f.write(
                f"[{format_time(seg['start'])} - {format_time(seg['end'])}] {seg['speaker']}: {seg['text']}\n"
            )

    print(f"Results saved to {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Whisper and Pyannote Audio Processing"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./data/ookawa.mp4",
        help="Path to the input MP4 file.",
    )
    parser.add_argument(
        "--wav_file",
        type=str,
        default="./data/ookawa_edited.wav",
        help="Path to the output WAV file.",
    )
    parser.add_argument(
        "--model_size", type=str, default="large-v2", help="Size of the Whisper model."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for model inference.",
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default="int8_float16",
        choices=["int8", "float16", "int8_float16"],
        help="Compute type for the model.",
    )
    parser.add_argument(
        "--language", type=str, default="ja", help="Language of the audio."
    )
    parser.add_argument(
        "--faster",
        action="store_true",
        default=False,
        help="Use faster-whisper for transcription.",
    )
    parser.add_argument(
        "--num_speakers",
        type=int,
        default=None,
        help="Number of speakers for diarization.",
    )
    parser.add_argument(
        "--output_file", type=str, default="output.txt", help="Output file name."
    )
    args = parser.parse_args()

    main(args)
