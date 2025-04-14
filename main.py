from subtitles import create_subtitle_video
from editor import TikTokEditor

import torch
import subprocess
import threading
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser(description="AI-video-editor")
parser.add_argument("--video-path", type=str, default="", help="Path to video to be edited")
parser.add_argument("--max-num-faces", type=int, default=2, help="Maximum number of faces in the video")
parser.add_argument("--n-speakers", type=int, default=2, help="number of speakers in the video")
parser.add_argument("--show-video", action="store_true", help="Shows video while doing face detection/embeddings")

args = parser.parse_args()

# Check if cuda is available
if torch.cuda.is_available():
    print("🚀 CUDA/GPU is being used for processing.")
else:
    print("⚡ CUDA/GPU is not available, running on CPU.")

os.makedirs("outputvids", exist_ok=True)
os.makedirs("segments_cache", exist_ok=True)
os.makedirs("subtitles_cache", exist_ok=True)

video_path = args.video_path
video_title = Path(video_path).stem

editor = TikTokEditor(video_path, args.n_speakers, args.max_num_faces, args.show_video)

# Launch GUI in a background thread
worker = threading.Thread(target=editor.match_faces_to_voices)
worker.start()
editor.gui.launch_gui(editor.speaker_segments)
worker.join()

output_path = os.path.join("outputvids", "output.mp4")
output_final_path = os.path.join("outputvids", "output_final.mp4")
output_final_subtitled_path = os.path.join("outputvids", f"{video_title}_final_subtitled.mp4")
subtitle_path = os.path.join("subtitles_cache", f"{video_title}.srt")

editor.crop_video_on_speaker_bbox_static(output_path)
editor.extract_audio_and_apply_to_video(output_path, output_final_path)
os.remove(output_path)
create_subtitle_video(output_final_path, subtitle_path, output_final_subtitled_path)
os.remove(output_final_path)

# subprocess.run(["mpv", output_final_subtitled_path])