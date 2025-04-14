import ffmpeg
import whisper
import torch
import os
import shutil

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


def create_subtitle_video(video_path, srt_path, output_video_path):
    generate_word_srt(video_path, srt_path)
    add_subtitles_from_srt(video_path, srt_path, output_video_path)
