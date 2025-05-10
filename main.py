import torch
import argparse

from editor import TikTokEditor

parser = argparse.ArgumentParser(description="AI-video-editor")
parser.add_argument("--video-path", type=str, default="", help="Path to video to be edited")
parser.add_argument("--n-speakers", type=int, default=2, help="number of speakers in the video")
parser.add_argument("--threshold", type=float, default=0.4, help="Threshold for cosine similarity of face embeddings")
parser.add_argument("--add-subtitles", action="store_true", help="Add subtitles to output video")
parser.add_argument("--word-timestamps", action="store_true", help="Creates subtitles by word instead of by sentence")
parser.add_argument("--new-subs", action="store_true", help="Redo subtitles (set if replacing word subs with sentence subs or vice versa)")
parser.add_argument("--edit", action="store_true", help="Edit video without exporting to blender for further editing")
parser.add_argument("--delete-cache", action="store_true", help="Delete cache for input video and start fresh")

args = parser.parse_args()

# Check if cuda is available
if torch.cuda.is_available():
    print("CUDA/GPU is being used for processing.")
else:
    print("CUDA/GPU is not available, running on CPU.")

editor = TikTokEditor(args.video_path, args.n_speakers, args.threshold, args.word_timestamps, args.delete_cache)
editor.analyze()
editor.prepare_for_blender(args.add_subtitles, args.new_subs)
print("Successfully created cache file for importing to Blender")
print("Open Blender and run blender.py in Text Editor to create edited video")

if args.edit:
    if args.add_subtitles:
        editor.edit_w_subtitles()
    else:
        editor.edit()
