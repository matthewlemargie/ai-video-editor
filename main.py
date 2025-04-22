import torch
import subprocess
import threading
import argparse
from pathlib import Path
import os

from editor import TikTokEditor

parser = argparse.ArgumentParser(description="AI-video-editor")
parser.add_argument("--video-path", type=str, default="", help="Path to video to be edited")
parser.add_argument("--max-num-faces", type=int, default=2, help="Maximum number of faces in the video")
parser.add_argument("--n-speakers", type=int, default=2, help="number of speakers in the video")
parser.add_argument("--show-video", action="store_true", help="Shows video while doing face detection/embeddings")
parser.add_argument("--word-timestamps", action="store_true", help="Creates subtitles by word instead of by sentence")


args = parser.parse_args()

# Check if cuda is available
if torch.cuda.is_available():
    print("🚀 CUDA/GPU is being used for processing.")
else:
    print("⚡ CUDA/GPU is not available, running on CPU.")

editor = TikTokEditor(args.video_path, args.n_speakers, args.max_num_faces, args.show_video, args.word_timestamps)
editor.analyze()
editor.edit_w_subtitles()

subprocess.run(["mpv", editor.output_final_subtitled_path, "--volume=60"])
