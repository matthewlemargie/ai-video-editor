import ffmpeg
import whisper
import torch
import os
import shutil
from time import time

# Format SRT timestamp: 00:00:00,000
def format_srt_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"


# Extract audio from video
def extract_audio(video_path, audio_path):
    ffmpeg.input(video_path).output(audio_path, ac=1, ar='16000').run(overwrite_output=True)


# Generate word-by-word SRT
def generate_word_srt(video_path, srt_output):
    s = time()
    if os.path.exists(srt_output):
        return
    audio_path = "temp_audio.wav"

    print("[1] Extracting audio...")
    extract_audio(video_path, audio_path)

    print("[2] Transcribing with Whisper (word timestamps)...")
    model = whisper.load_model("medium", device="cuda" if torch.cuda.is_available() else "cpu")
    result = model.transcribe(audio_path, word_timestamps=True, verbose=False)

    print("[3] Writing word-level SRT...")
    index = 1
    with open(srt_output, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            for word in segment["words"]:
                start = format_srt_time(word["start"])
                end = format_srt_time(word["end"])
                text = word["word"].strip()

                if not text:
                    continue

                f.write(f"{index}\n{start} --> {end}\n{text}\n\n")
                index += 1

    os.remove(audio_path)
    print(f"[✓] Done! SRT file saved as: {srt_output}")
    print(f"Speech-to-text transcription with openai-whisper was completed in {time() - s:.2f}s")
    print()


def split_into_lines(text, max_chars):
    words = text.split()
    lines = []
    line = ""

    for word in words:
        if len(line + " " + word) <= max_chars:
            line += " " + word if line else word
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def generate_sentence_srt(video_path, srt_output):
    start = time()
    if os.path.exists(srt_output):
        return
    audio_path = "temp_audio.wav"

    print("[1] Extracting audio...")
    extract_audio(video_path, audio_path)

    print("[2] Transcribing with Whisper (word timestamps)...")
    model = whisper.load_model("medium", device="cuda" if torch.cuda.is_available() else "cpu")
    result = model.transcribe(audio_path, word_timestamps=True, verbose=False)
    MAX_CHARS = 30
    SENTENCE_ENDINGS = {".", "!", "?"}
    SMART_COMMA_SPLIT = True  # only split on comma if line is too long


    print("[3] Writing sentence-based SRT...")
    index = 1
    with open(srt_output, "w", encoding="utf-8") as f:
        buffer = ""
        sentence_start = None

        for segment in result["segments"]:
            for word in segment["words"]:
                word_text = word["word"].strip()
                if not word_text:
                    continue

                if buffer == "":
                    sentence_start = word["start"]

                buffer += word_text + " "
                word_end = word["end"]

                is_strong_end = word_text[-1] in SENTENCE_ENDINGS
                is_soft_comma = word_text[-1] == "," and len(buffer.strip()) > MAX_CHARS if SMART_COMMA_SPLIT else False

                if is_strong_end or is_soft_comma:
                    sentence = buffer.strip()

                    # Optional: wrap long lines
                    wrapped_lines = split_into_lines(sentence, MAX_CHARS)

                    f.write(f"{index}\n{format_srt_time(sentence_start)} --> {format_srt_time(word_end)}\n")
                    for line in wrapped_lines:
                        f.write(line + "\n")
                    f.write("\n")

                    index += 1
                    buffer = ""
                    sentence_start = None

        # Handle leftover buffer
        if buffer:
            sentence = buffer.strip()
            wrapped_lines = split_into_lines(sentence, MAX_CHARS)
            f.write(f"{index}\n{format_srt_time(sentence_start)} --> {format_srt_time(word_end)}\n")
            for line in wrapped_lines:
                f.write(line + "\n")
            f.write("\n")

    os.remove(audio_path)
    print(f"[✓] Done! SRT file saved as: {srt_output}")
    print(f"Speech-to-text transcription with openai-whisper was completed in {time() - start:.2f}s")
    print()


def add_subtitles_from_srt(video_path, srt_path, output_video_path):
    # Check if the input video and SRT file exist
    if not os.path.exists(video_path):
        print(f"[!] Video file not found: {video_path}")
        return
    if not os.path.exists(srt_path):
        print(f"[!] SRT file not found: {srt_path}")
        return

    if os.path.exists(srt_path):
        shutil.move(srt_path, os.path.basename(srt_path))

    print(f"[1] Adding subtitles from '{srt_path}' to '{video_path}' with CUDA acceleration...")

    try:
        # Use ffmpeg to overlay subtitles on the video with CUDA for encoding
        ffmpeg.input(video_path).output(output_video_path, vf=f"subtitles={os.path.basename(srt_path)}", 
                                        vcodec="h264_nvenc").run(overwrite_output=True)
        print(f"[✓] Done! Video saved as: {output_video_path}")
        if os.path.exists(os.path.basename(srt_path)):
            shutil.move(os.path.basename(srt_path), srt_path)
    except ffmpeg.Error as e:
        print(f"[!] Error occurred while processing: {e}")
